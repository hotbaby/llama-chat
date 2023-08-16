# llama-chat

Chat with Llama via Web.



## 下载模型

到[hugginface.co](https://huggingface.co/meta-llama)下载对应版本的LLaMA模型weights，保存到`/data/models/llama`目录下。



## 本地运行

安装依赖包

```bash
pip install -r requirements.txt
```

运行

```bash
streamlit run web.py 
```



## 容器化运行

构建镜像

```bash
docker build . -t llama-chat
```

运行容器

```bash
docker run -d --rm --name llama-chat -v /data:/data -p 8501:8501 llama-chat
```

> 如果前面使用nginx进行返现代理，需要关闭CORS和XSRF。



## Llama微调系统模型

```bash
# 下载DoctorGPT模型参数
git clone git@hf.co:llSourcell/medllama2_7b /data/models/medllama2_7b

# 构建llama-chat镜像
git clone git@github.com:hotbaby/llama-chat.git
cd llama-chat
docker build . -t llama-chat

# 运行Web服务
# MODEL_NAME环境变量指定模型名称。
# MODEL_PATH环境变量指定模型参数路径。
# 服务导出端口是8501。
docker run -it -d --rm --name doctor-gpt -v /data:/data/ -p 8501:8501 -e MODEL_NAME=DoctorGPT -e MODEL_PATH=/data/models/medllama2_7b llama-chat
```