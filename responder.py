from chat_ai import Network
def response(text):
    net=Network()
    net.load_datas()
    model=net.create_model()
    net.load_w(model)
    return net.ez_response(text)