from chat_ai import Network
def api(text):
    net=Network()
    net.load_datas()
    model=net.create_model()
    net.load_w(model)
    return net.ez_response(text)
if __name__ == "__main__":
    net=Network()
    net.load_datas()
    model=net.create_model()
    net.load_w(model)
    while True:
        key=input('>')
        if key:
            
            try:
                response=net.ez_response(key)
            
            except:
                response='error'
            print(response)
        else:
            break