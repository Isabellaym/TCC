from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2


class Emocoes():
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.imagem = None
        self.face_cascade = cv2.CascadeClassifier('haarcascade_faces.xml')
        self.model = load_model('rede_neural_peso.h5')
        self.emocoes = ["Raiva", "Desgosto", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]

    def encerra_execucao(self, mensagem):
        print(mensagem)
        self.video.release()
        cv2.destroyAllWindows()
        exit(0)

    def run(self):
        while True:

            conectado, self.imagem = self.video.read()

            if not conectado:
                self.encerra_execucao("Não foi possível capturar a imagem")

            imagem_cinza = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(imagem_cinza, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                for (x, y, largura, altura) in faces:
                    cv2.rectangle(self.imagem, (x, y), (x + largura, y + altura), (0, 255, 0), 1)
                    roi = imagem_cinza[y:y + altura, x:x + largura]
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    emocoes = self.model.predict(roi)[0]

                    if emocoes is not None:
                        probabilidade = np.argmax(emocoes)
                        cv2.putText(self.imagem, self.emocoes[probabilidade], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Deteccao de Emocao", self.imagem)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.encerra_execucao('Execução do programa encerrada')


if __name__ == '__main__':
    deteccao_emocoes = Emocoes()
    deteccao_emocoes.run()
