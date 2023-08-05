from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
import jsonlines

EMBEDDING_DIR = '/data/wusiyao/chatglm/chatglm/text2vec-large-chinese'

class JsonlLoader(BaseLoader):
    def __init__(self, filepath):
        self.filepath = filepath
    def load(self):
        docs = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for i, item in enumerate(jsonlines.Reader(f)):
                text = item[0].strip()
                metadata = dict(
                    source = str(self.filepath),
                    seq_num = i
                )
                docs.append(Document(page_content=text, metadata=metadata))
        return docs

class DocumentService(object):
    def __init__(self, docs_path, vector_store_path):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_DIR)
        self.docs_path = docs_path
        self.vector_store_path = vector_store_path
        self.vector_store = None

    def init_source_vector(self):
        """
        初始化本地知识库向量
        :return:
        """
        # 读取文本文件
        docs = JsonlLoader(self.docs_path).load()
        # 采用embeding模型对文本进行向量化
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        # 把结果存到faiss索引里面
        self.vector_store.save_local(self.vector_store_path)

    def load_vector_store(self):
        self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
