# Use AWS Lambda base image for Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Save environment variables from build arguments
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ARG AWS_DEFAULT_REGION
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
ARG DYNAMODB_TABLE
ENV DYNAMODB_TABLE=${DYNAMODB_TABLE}

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
    fmtutil-sys --all

# Install Python dependencies
COPY tensorgrad/ tensorgrad/
COPY docker/ docker/
RUN pip install ./docker

# Command to start the Lambda function
CMD ["docker.drawTensors.lambda_handler"]
