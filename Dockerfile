# ---- LibreDWG command line tools (dwg2dxf / dxf2dwg) for DWG import/export
FROM python:3.11-slim AS libredwg
ARG LIBREDWG=0.13.3
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl xz-utils ca-certificates \
 && curl -fsSL -o /tmp/l.tar.xz https://github.com/LibreDWG/libredwg/releases/download/${LIBREDWG}/libredwg-${LIBREDWG}.tar.xz \
 && tar -C /tmp -xf /tmp/l.tar.xz && cd /tmp/libredwg-${LIBREDWG} \
 && ./configure --disable-bindings --disable-docs --prefix=/opt/libredwg && make -j"$(nproc)" && make install

# ---- application
FROM python:3.11-slim
COPY --from=libredwg /opt/libredwg /opt/libredwg
ENV PATH="/opt/libredwg/bin:${PATH}" LD_LIBRARY_PATH="/opt/libredwg/lib"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY floorplan ./floorplan
COPY samples ./samples
EXPOSE 8000
CMD ["uvicorn", "floorplan.server:app", "--host", "0.0.0.0", "--port", "8000"]
