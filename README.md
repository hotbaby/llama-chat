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

