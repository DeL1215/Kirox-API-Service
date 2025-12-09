from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print(model)
print(model.get_sentence_embedding_dimension())  
vec = model.encode(["hello world"])[0]
print(len(vec))   
print("模型裝置：", model.device)
