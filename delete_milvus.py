from pymilvus import Collection, connections, list_collections
connections.connect(alias="default", host="127.0.0.1", port="19530")
print(list_collections())
# 若有舊 Collection，直接刪
Collection("chat_memory").drop()
Collection("kb_memory").drop()

