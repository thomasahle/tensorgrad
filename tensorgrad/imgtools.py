from pdf2image import convert_from_path

import os
import subprocess
from PIL import Image, ImageDraw

from tensorgrad.serializers.to_tikz import to_tikz

from tensorgrad import Derivative
from tensorgrad.extras import Expectation

import networkx as nx


def compile_latex(latex_code, suffix=""):
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)

    # Save the LaTeX code to a file
    tex_file_path = os.path.join(output_dir, f"output_{suffix}.tex")
    with open(tex_file_path, "w") as file:
        file.write(latex_code)

    # Compile the LaTeX file to PDF
    subprocess.run(
        ["lualatex", "-output-directory", output_dir, tex_file_path],
        check=True,
    )

    # Convert the PDF to an image
    pdf_path = tex_file_path.replace(".tex", ".pdf")
    images = convert_from_path(pdf_path, dpi=300)

    # Save the first page as an image
    image_path = pdf_path.replace(".pdf", ".png")
    images[0].save(image_path, "PNG")

    return image_path


def combine_images_vertically(
    image_paths, padding=10, line_padding=5, background_color="white", line_color="black", line_width=2
):
    images = [Image.open(x) for x in image_paths]

    # Calculate total height considering padding, maximum width, separating lines, and line padding
    total_height = (
        sum(image.height for image in images)
        + padding * (len(images) + 1)
        + line_width * (len(images) - 1)
        + line_padding * 2 * (len(images) - 1)
    )
    max_width = max(image.width for image in images) + padding * 2

    # Create a new image with a white background
    combined_image = Image.new("RGB", (max_width, total_height), color=background_color)

    # Create a drawing context
    draw = ImageDraw.Draw(combined_image)

    # Paste images into the new image with padding, separating lines, and line padding
    y_offset = padding
    for i, image in enumerate(images):
        # Calculate horizontal padding to center images
        x_padding = (max_width - image.width) // 2

        combined_image.paste(image, (x_padding, y_offset))
        y_offset += image.height + padding

        # Add line padding below the current image
        y_offset += line_padding

        # Draw a black separating line below the current image (except for the last image)
        if i < len(images) - 1:
            line_y = y_offset + line_width // 2
            draw.line([(0, line_y), (max_width, line_y)], fill=line_color, width=line_width)
            y_offset += line_width + line_padding

    # Save the combined image
    combined_image_path = "combined_image.png"
    combined_image.save(combined_image_path)

    return combined_image_path


def save_as_image(expr, path):
    latex_code = to_tikz(expr)
    image_path = compile_latex(latex_code)
    os.rename(image_path, path)
    print(f"Image saved to {path}")


def save_steps_old(expr, min_steps=None):
    images = []
    images.append(compile_latex(expr, suffix="0"))
    old = expr
    while True:
        new = expr.simplify({"grad_steps": len(images)})
        if new == old and (min_steps is None or len(images) >= min_steps):
            print(f"{new=} = {old=}")
            break
        old = new
        images.append(compile_latex(to_tikz(new), suffix=f"{len(images)}"))

    output_path = combine_images_vertically(images)
    print(f"Combined image saved to {output_path}")


def save_steps(expr, slow_grad=False):
    images = []
    images.append(compile_latex(to_tikz(expr), suffix="0"))

    # TODO: This is still not good enough, since there may be a double derivative somewhere inside the tensor.
    cnt_derivatives = 0
    d = expr
    while isinstance(d, Derivative) or isinstance(d, Expectation):
        cnt_derivatives += 1
        d = d.tensor
    if cnt_derivatives > 1:
        expr = expr.simplify({"grad_steps": cnt_derivatives})
        images.append(compile_latex(to_tikz(expr), suffix=f"{len(images)}"))

    expand = False
    while True:
        try:
            args = {"grad_steps": 1} if slow_grad else {}
            if expand:
                args["expand"] = True
            new = expr.simplify(args).simplify()
        except Exception as e:
            print("ERROR:", e)
            break
        if new == expr:
            if not expand:
                expand = True
                continue
            break
        print(new)
        images.append(compile_latex(to_tikz(new), suffix=f"{len(images)}"))
        expr = new

    output_path = combine_images_vertically(images)
    print(f"Combined image saved to {output_path}")


def draw_structural_graph(tensor, iter=50):
    G, edges = tensor.structural_graph()
    for e, node in edges.items():
        n = G.number_of_nodes()
        G.add_node(n, name=f"{e}")
        G.add_edge(n, node)
    labels = {i: data.get("name", "") for i, data in G.nodes(data=True)}
    pos = nx.spectral_layout(G)
    pos = nx.kamada_kawai_layout(G, pos=pos)
    pos = nx.spring_layout(G, pos=pos, k=1, iterations=iter)

    nx.draw_networkx(G, pos=pos, labels=labels)
