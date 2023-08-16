FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /opt/apps/DoctorGPT

COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD streamlit run --server.address 0.0.0.0 --server.port 8501 --server.enableCORS=false --server.enableXsrfProtection=false web.py
