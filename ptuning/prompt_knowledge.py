from transformers import AutoModel, AutoTokenizer
import json, re
from tqdm import trange
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
import jsonlines

EMBEDDING_DIR = '/data/wusiyao/chatglm/chatglm/text2vec-large-chinese'
DOCS_PATH = '/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/AFAC_LLM/rawdata/观点库_合并330.jsonl'
VS_PATH = '/data/wusiyao/chatglm/chatglm/sourcecode/knowledge_base'
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

PROMPT = """
你是一个资深金融领域基金研究员，请根据已知信息和基金结构化信息，提供一条基金推荐话术。
已知信息：
<{knowledge}>

基金结构化信息：<{info}>
基金推荐话术：
"""


if __name__ == "__main__":
    device = 'cuda:1'
    model = AutoModel.from_pretrained('/data/wusiyao/chatglm/chatglm/chatglm2-6b', trust_remote_code=True).half().to(
        device)
    tokenizer = AutoTokenizer.from_pretrained('/data/wusiyao/chatglm/chatglm/chatglm2-6b', trust_remote_code=True)
    model = model.eval()
    # 本地知识库
    vs_path = '/data/wusiyao/chatglm/chatglm/sourcecode/knowledge_base'
    know = DocumentService(docs_path='/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/AFAC_LLM/rawdata/观点库.jsonl',
                           vector_store_path=vs_path)
    # 初始化向量库
    # know.init_source_vector()
    # 加载向量库
    know.load_vector_store()
    vectore_store = know.vector_store

    queries = []
    with open('/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/test.json', 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                break
            queries.append(json.loads(line)['text'])

    with open('/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/out/prompt_chatglm2/prompt_knowledge.json', 'w+',
              encoding='utf-8') as f:

        for i in trange(600):
            zhi_q = queries[i * 2]
            lihao_q = queries[i * 2 + 1]
            result_info, result_event = [], []
            related_docs_zhi = vectore_store.similarity_search_with_score(zhi_q, k=1)[0][0].page_content  # 查找相关信息
            related_docs_lihao = vectore_store.similarity_search_with_score(lihao_q, k=1)[0][0].page_content  # 查找相关信息

            for _ in range(5):
                response, _ = model.chat(tokenizer,
                                         prompt.replace("{info}", zhi_q).replace("{knowledge}", related_docs_zhi),
                                         history=[])
                result_info.append(response)

            for _ in range(5):
                response, _ = model.chat(tokenizer,
                                         prompt.replace("{info}", lihao_q).replace("{knowledge}", related_docs_lihao),
                                         history=[])
                result_event.append(response)

            row = {'id': str(i), "info_result": result_info, "event_result": result_event}
            f.write(json.dumps(row, ensure_ascii=False))
