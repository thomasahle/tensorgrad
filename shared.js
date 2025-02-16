function dedent(str) {
   const lines = str.split('\n');
   if (lines[0].trim() === '') lines.shift();
   if (lines[lines.length - 1].trim() === '') lines.pop();
   const indentLengths = lines
      .filter(line => line.trim().length > 0)
      .map(line => line.match(/^(\s*)/)[1].length);
   const minIndent = Math.min(...indentLengths);
   return lines.map(line => line.slice(minIndent)).join('\n');
}

// Updated examples object using dedent:
const examples = [
   {
      title: "Product Derivative",
      code: dedent(`
      # Define sizes for the tensor edges and variables
      i, j, k, l = sp.symbols("i j k l")
      T = tg.Variable("T", i, j, k)
      M = tg.Variable("M", k, l)

      # Define the expression and its derivative wrt M
      P = T @ M  #
      expr = tg.Derivative(P, M)

      # Simplify the expression and save the steps
      save_steps(expr)

      # Press "Run" to see the result!
    `)
   },
   {
      title: "Derivative of ‖ AᵀA − I ‖_F²",
      code: dedent(`
      # Define A to be a square matrix
      i = sp.symbols("i")
      A = tg.Variable("A", i, j=i)

      # Compute A.T @ A - I
      AT = A.rename(j='k')
      I = tg.Delta(i, 'j', 'k')
      error = AT @ A - I

      # Contract with itself to take Frobenius norm square
      frob = error @ error

      # Take the derivative with respect to A
      expr = tg.Derivative(frob, A)

      # Simplify the expression and save the steps
      save_steps(expr)
    `)
   },
   {
      title: "Derivative of L2 Loss",
      code: dedent(`
      # Define sizes for the tensor edges and variables
      b, x, y = sp.symbols("b x y")
      X = tg.Variable("X", b, x)
      Y = tg.Variable("Y", b, y)
      W = tg.Variable("W", x, y)

      # Define the error
      error = X @ W - Y

      # The Frobenius norm squared is just the contraction
      # of the all the edges of the tensor with itself
      loss = error @ error

      # Compute the derivative of the loss wrt W
      expr = tg.Derivative(loss, W)

      # Simplify the expression and save the steps
      save_steps(expr)
    `)
   },
   {
      title: "Hessian of CE Loss",
      code: dedent(`
      # Define the logits and targets as vectors
      i = sp.symbols("i")
      logits = tg.Variable("logits", i)
      target = tg.Variable("target", i)

      # Define the softmax cross-entropy loss
      e = F.exp(logits)
      softmax = e / F.sum(e)
      ce = -F.sum(target * F.log(softmax))

      # Compute the Hessian by taking the gradient of the gradient
      H = ce.grad(logits).grad(logits)

      # Simplify the expression and save the steps
      save_steps(H.full_simplify())
    `)
   },
   {
      title: "Expectation of L2 Loss",
      code: dedent(`
      # Define sizes for the tensor edges and variables
      b, x, y = sp.symbols("b x y")
      X = tg.Variable("X", b, x)
      Y = tg.Variable("Y", b, y)
      W = tg.Variable("W", x, y)

      # Define the mean and covariance variables of the distribution
      mu = tg.Variable("mu", x, y)
      C = tg.Variable("C", x, y, x2=x, y2=y)

      # Take the expectation of the L2 loss if W was Gaussian
      XWmY = X @ W - Y
      l2 = F.sum(XWmY * XWmY)
      E = tg.Expectation(l2, W, mu, C)

      # Simplify the expression and save the steps
      save_steps(E.full_simplify())
    `)
   },
   {
      title: "Isserlis' Theorem",
      code: dedent(`
      # Isserlis' Theorem tells us that E[u ⊗ u ⊗ u ⊗ u], where u is Gaussian
      # with mean 0 and covariance matrix C, is "roughly" 3 C ⊗ C, but symmetrized

      # Define u as a vector
      i = sp.symbols("i")
      u = tg.Variable(f"u", i)

      # Take the tensor product
      prod = tg.Product([u.rename(i=f"i{k}") for k in range(4)])

      # Define the symmetric covariance matrix
      C = tg.Variable(f"C", i, j=i).with_symmetries("i j")

      # Take the expectation of 'prod' wrt u
      expr = tg.Expectation(prod, u, mu=tg.Zero(i), covar=C, covar_names={"i": "j"})
      save_steps(expr.full_simplify())
      `)
   },
   {
      title: "Tensor Taylor Approximation",
      code: dedent(`
      # Compute the Taylor Approximation
      # softmax(x + eps) = softmax(x) + eps * ...

      # First define two vectors of the same shape
      i = sp.symbols("i")
      x = tg.Variable("x", i)
      eps = tg.Variable("eps", i)

      # Take the softmax and expand it in terms of simple functions
      y = F.softmax(x, dim='i').simplify({'expand_functions': True})

      # Take the second order tensor approximation of Y wrt (x, eps)
      expr = F.taylor(y, x, eps, n=2)

      save_steps(expr.full_simplify())
      `)
   },
];

// Server endpoints (adjust to your environment)
const SERVER_BASE_URL = "https://4kqy5zmzdi3aghjn32orugt7vm0kgzts.lambda-url.us-east-1.on.aws";
const EXECUTE_ENDPOINT = `${SERVER_BASE_URL}/execute`;
const SNIPPETS_ENDPOINT = `${SERVER_BASE_URL}/snippets`;
const FORMAT_ENDPOINT = `${SERVER_BASE_URL}/format`;


async function initializePage(config) {
   const runBtn = document.getElementById(config.runBtnId);
   const outputDiv = document.getElementById(config.outputId);
   const errorDiv = document.getElementById(config.errorId);
   const imageContainer = document.getElementById(config.imageContainerId);
   const formatBtn = document.getElementById(config.formatBtnId);
   const shareBtn = document.getElementById(config.shareBtnId);
   const exampleSelect = document.getElementById(config.exampleSelectId);
   const snippetInput = document.getElementById(config.snippetInputId);
   const editor = CodeMirror.fromTextArea(document.getElementById(config.textareaId), {
      mode: "python",
      lineNumbers: true
   });

   function setSnippet(snippetParam) {
      if (snippetParam) {
         if (snippetInput) {
            snippetInput.style.display = "block";
            snippetInput.value = snippetParam;
            snippetInput.select();
            snippetInput.scrollLeft = snippetInput.scrollWidth;
         }
         if (exampleSelect) {
            exampleSelect.style.display = "none";
         }
      }
      else {
         if (snippetInput) {
            snippetInput.style.display = "none";
         }
         if (exampleSelect) {
            exampleSelect.style.display = "block";
            exampleSelect.style.visibility = "visible";
         }
      }
   }

   /**
    * Run code by calling /execute. Displays output or error in given elements.
    */
   // 2. Wire up "Run" button
   if (runBtn) {
      runBtn.addEventListener("click", async function() {
         // Clear old output/error
         if (errorDiv) {
            errorDiv.style.display = "none";
            errorDiv.textContent = "";
         }

         // Show loading (if button exists)
         runBtn.disabled = true;
         runBtn.innerHTML = '<div class="loading"></div>Running...';

         try {
            const code = editor.getValue();
            const resp = await fetch(EXECUTE_ENDPOINT, {
               method: "POST",
               headers: { "Content-Type": "application/json" },
               body: JSON.stringify({ code })
            });
            console.log(resp);

            if (!resp.ok) {
               errorDiv.textContent = `Server returned status ${resp.status}`;
               errorDiv.style.display = "block";
               return;
            }

            const data = await resp.json();
            if (!data.success) {
               let msg = data.error || "Unknown error.";
               if (data.stacktrace) {
                  msg += "\n\n" + data.stacktrace;
               }
               // Display error message
               errorDiv.textContent = msg;
               errorDiv.style.display = "block";
               return;
            }

            // Display image (if available)
            if (data.image) {
               const img = document.createElement('img');
               img.src = data.image;
               img.alt = 'Tensor visualization';
               imageContainer.innerHTML = '';
               imageContainer.appendChild(img);
            } else {
               console.log("No image returned.");
               return;
            }
         } finally {
            // Set button back to normal
            runBtn.disabled = false;
            runBtn.innerHTML = "Run";
         }
      });
   }

   // 3. Wire up "Share" button
   /**
    * Share snippet: POST code to /snippets, then redirect to playground.html?snippet={uuid}
    */
   if (shareBtn) {
      shareBtn.addEventListener("click", async () => {
         // Show loading (if button exists)
         shareBtn.disabled = true;
         shareBtn.innerHTML = '<div class="loading"></div>Sharing...';

         try {
            const code = editor.getValue();
            const resp = await fetch(SNIPPETS_ENDPOINT, {
               method: "POST",
               headers: { "Content-Type": "application/json" },
               body: JSON.stringify({ code })
            });
            console.log(resp);
            if (!resp.ok) {
               throw new Error(`Server returned status ${resp.status}`);
            }
            const data = await resp.json();
            console.log(data);

            // If we're already on playground.html, update the URL and input field.
            const urlWithoutQuery = window.location.href.split('?')[0];
            if (urlWithoutQuery.endsWith("playground.html")) {
               const url = new URL(window.location.href);
               url.searchParams.set("snippet", data.snippet_id);
               history.pushState(null, "", url); // update the URL without a page reload
               console.log("Updated URL to", url.toString());
               // Update the snippet input field (if configured) and select its content.
               // Also hide the examples dropdown if needed.
               setSnippet(url.toString());
            } else {
               console.log("Redirecting to playground.html?snippet=" + data.snippet_id);
               window.location.href = `playground.html?snippet=${data.snippet_id}`;
            }
         } finally {
            // Set button back to normal
            shareBtn.disabled = false;
            shareBtn.innerHTML = "Share";
         }
      });
   }

   // 4. Wire up "Format" button
   if (formatBtn) {
      formatBtn.addEventListener("click", async function() {
         if (errorDiv) {
            errorDiv.style.display = "none";
            errorDiv.textContent = "";
         }

         if (formatBtn) {
            formatBtn.disabled = true;
            formatBtn.innerHTML = '<div class="loading"></div>Formatting...';
         }

         try {
            const code = editor.getValue();
            const resp = await fetch(FORMAT_ENDPOINT, {
               method: "POST",
               headers: { "Content-Type": "application/json" },
               body: JSON.stringify({ code })
            });
            console.log(resp);
            if (!resp.ok) {
               throw new Error(`Server returned status ${resp.status}`);
            }

            const data = await resp.json();
            if (!data.success || !data.formattedCode) {
               throw new Error(data.error || "Formatting failed.");
            }

            editor.setValue(data.formattedCode);
         } catch (err) {
            if (errorDiv) {
               errorDiv.textContent = "Error: " + err.message;
               errorDiv.style.display = "block";
            }
         } finally {
            if (formatBtn) {
               formatBtn.disabled = false;
               formatBtn.innerHTML = "Format";
            }
         }
      });
   }

   // 5. Populate and wire up "Examples" dropdown
   if (exampleSelect) {
      // Clear any existing options.
      exampleSelect.innerHTML = "";
      // Create an option for each example.
      for (const [key, example] of examples.entries()) {
         const option = document.createElement("option");
         option.value = key;
         option.text = examples[key].title;
         exampleSelect.appendChild(option);
      }
      // When the user selects an example, load its code.
      exampleSelect.addEventListener("change", () => {
         const key = exampleSelect.value;
         if (key && examples[key]) {
            editor.setValue(examples[key].code);
         }
      });
      // Set editor content to first example.
      editor.setValue(examples[0].code);
   }

   // Load snippet from /snippets/{snippetId} and set the editor content.
   // 6. Auto-load snippet from URL if requested.
   if (config.autoLoadSnippetFromURL) {
      const params = new URLSearchParams(window.location.search);
      const snippetParam = params.get("snippet");
      if (snippetParam) {

         console.log("Loading snippet", snippetParam);

         const resp = await fetch(`${SNIPPETS_ENDPOINT}/${snippetParam}`);
         console.log(resp);
         if (!resp.ok) {
            throw new Error(`Snippet not found or server error (status ${resp.status})`);
         }
         const data = await resp.json();
         if (!data.code) {
            throw new Error("No code found in snippet.");
         }
         editor.setValue(data.code);
         // Show the snippet input field and hide the examples dropdown.
         setSnippet(window.location.href);
      }
      else {
         setSnippet(null);
      }
   }
}
