import requests
import time
import threading
from pynput.keyboard import Key, Controller


URL_POP = 'https://3vkeycenej.execute-api.us-east-1.amazonaws.com/prod/CIAB-2018-DroneCommandQueue?droneId=drone1&operation=POP&secret=dr0neRulez2A5T7U'
URL_PEEK = 'https://3vkeycenej.execute-api.us-east-1.amazonaws.com/prod/CIAB-2018-DroneCommandQueue?droneId=drone1&operation=PEEK&secret=dr0neRulez2A5T7U'

URL_PUSH_STATUS = 'https://3vkeycenej.execute-api.us-east-1.amazonaws.com/prod/CIAB-2018-DroneCommandQueue?operation=PUSH&secret=dr0neRulez2A5T7U&droneId=status&command='

class Servidor():
    def __init__(self):
        self.cancelar_prepara = False
        self.comando = "empty"
        self.cancelarConexao = False
        self.executando = False
        self.droneConectado = False

    def setExecutando(self, value):
        self.executando = value;

    def solicitar_conexao(self):
        print("aguardando comando...")
        #joga fora o primeiro comando que estiver na pilha
        r = requests.get(URL_POP)

        #Faz uma solicitação no servidor por segundo, até o comando não
        #ser mais empty
        r = requests.get(URL_POP)
        while(r.text=='empty'):
            #Verifica se a conexão não foi cancelada pela outra thread, cancelarConexao
            if(self.cancelarConexao==True):
                return

            #Verifica se não houve perda de conexão
            for cf in self.mTeste.cfs:
                if (cf.link == None):
                    print("conexao com o drone perdida")
                    self.cancelarConexao = True
                    self.droneConectado = False

            r = requests.get(URL_POP)
            time.sleep(1)

        self.comando = r.text
        Servidor.press_enter()

    def cancela_comando(self):
        entrada = input("tecle ENTER se quiser cancelar: ")
        while(not self.cancelarConexao):
            if(entrada==""):
                self.cancelarConexao = True
                return


    def verificar_comando(self, mTeste):
        self.mTeste = mTeste
        threadComando = threading.Thread(target=self.solicitar_conexao)
        threadComando.setDaemon(True)

        threadCancela = threading.Thread(target=self.cancela_comando)
        threadCancela.setDaemon(True)

        threadComando.start()
        threadCancela.start()

        threadComando.join()



    def verificar_prepara(self):
        threadEsperaPrepara = threading.Thread(target=self.espera_prepara)
        threadEsperaPrepara.setDaemon(True)

        threadContinua = threading.Thread(target=self.cancela_prepara)
        threadContinua.setDaemon(True)

        threadEsperaPrepara.start()
        threadContinua.start()

        threadEsperaPrepara.join()



    def espera_prepara(self):
        print("\n===============================================================================\n"
              "Aguardando prepara...")
        #joga fora o primeiro comando da pilha
        r = requests.get(URL_POP)

        r = requests.get(URL_POP)
        while(r.text!= "preparar"):
            if(self.cancelar_prepara):
                return
            r = requests.get(URL_POP)
            time.sleep(1)
        Servidor.press_enter()


    def cancela_prepara(self):
        while(not self.cancelar_prepara):
            entrada = input("tecle ENTER se quiser continuar: "
                            "\n===============================================================================\n")
            if(entrada==""):
                self.cancelar_prepara = True
                break


    def push_status(command):
        r = requests.get(URL_PUSH_STATUS + command)

    def press_enter():
        #Simula um enter do teclado: utilizado para o usuário não precisar ficar
        # dando enter desnecessariamente
        keyboard = Controller()
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
