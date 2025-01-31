# Use AWS Lambda base image for Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Install necessary Linux packages (TeX Live, Poppler)
# RUN microdnf update -y \
#     && microdnf install -y \
#         texlive-luatex \
#         texlive-scheme-basic \
#         poppler-utils \
#     && microdnf clean all \
#     && rm -rf /var/cache/dnf
# 
# RUN tlmgr update --self \
#     && tlmgr install \
#     scheme-basic \
#     luatex \
#     pgf \
#     standalone \
#     comicneue \
#     geometry \
#     graphics \
#     latex-bin


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
