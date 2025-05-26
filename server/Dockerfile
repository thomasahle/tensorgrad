# Use AWS Lambda base image for Python 3.12
FROM public.ecr.aws/lambda/python:3.12

ENV AWS_DEFAULT_REGION="us-east-1"
ENV DYNAMODB_CACHE_TABLE="TensorgradCache"
ENV DYNAMODB_SNIPPET_TABLE="CodeSnippets"

# Install microdnf for lightweight package management
RUN microdnf update -y \
   && microdnf install -y \
      glibc-langpack-en \
      texlive-scheme-basic \
      texlive-standalone \
      texlive-luatex85 \
      texlive-comicneue \
      poppler-utils \
   && microdnf clean all

# Set TEXMFVAR to a writable location
ENV TEXMFVAR=/tmp

# Pre-generate format files during build
RUN mkdir -p /tmp && \
    fmtutil-sys --all && \
    luaotfload-tool --update

# Install Python dependencies
COPY tensorgrad/ tensorgrad/
COPY server/ server/
RUN pip install ./server

# Command to start the Lambda function
CMD ["server.drawTensors.handler"]
