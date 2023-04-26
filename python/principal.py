#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 22:18:24 2021

@author: Ítalo Santos
"""

import sys #Para utilizar algo que dependa do sistema
import os
import shutil
import cv2 #Para abrir e salvar imagens
import numpy as np #Para manipulação vetorial
import matplotlib.pyplot as plt #Para plotar gráficos
import copy #Para poder copiar o conteúdo de uma listas de objetos

from PyQt5.QtWidgets import QMainWindow, QApplication, qApp, QFileDialog, QMessageBox, QInputDialog #Para construção e funcionamento da interface
from PyQt5.QtGui import QPixmap, QImage, qRgb, QPainter, QPen, QColor #Para poder manipular as matrizes em imagens que serão mostradas na interface
from PyQt5.QtCore import Qt, QPointF, QLineF, QRectF #Para formatação da imagem na interface

#Importando todas as classes da interface
from gui_italo_interface import *


#Objetos dessa classe vão ter armazenar a matriz de valores (ou tensor se tiver mais de um canal) e o modelo de cor (RBG,Mono,Bin,HSB)
class Imagem():
    def __init__(self): #Cada objeto criado começa sem nenhuma identificação
        #Guarda matriz ou matrizes de valores de cada canal
        self.matriz = None
        #Tipo 1: RGB | Tipo 2: Mono | Tipo 3: Bin
        self.tipo = None
        #Guarda resultados de instensidade do detector de bordas
        self.bordasAbs = None
        #Guarda resultados sentido das bordas em termos de ângulo em radianos
        self.bordasSentido = None
        #Guarda vetor ou matriz de Histograma
        self.histograma = None
        #Guarda regiões
        self.mapaRegioes = None
        #Guarda quantidade de regiões
        self.numRegioes = 0
        #Guarda vetor de características de cada região
        self.regioes = None

class Regiao():
    def __init__(self,label,mapa):
        #Para identificação e localização da região
        self.label = label
        self.mapa = mapa
        #Características
        self.area = None
        self.centroDeMassa = None
        self.orientacao = None
        self.comprimento = None
        self.largura = None
        #Usado para plotar reta com orientação
        self.extremos = None

class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #self.setWindowTitle('Visão Computacional')
        self.ui.botaoUseResultado.clicked.connect(self.usarResultado)
        self.ui.botaoMoveParaAuxiliar.clicked.connect(self.moverParaAuxiliar)
        self.im1 = Imagem() #Cria um campo para im original
        self.imRes = Imagem() #Cria um campo para im processada
        self.im2 = Imagem() # Cria um campo para im auxiliar para operações com duas imagens
        self.imAux = Imagem() #Cria um campo para im extra
        self.sliderflag = 0 #Cria uma variável de uso para slider
        self.ui.menubar.setNativeMenuBar(False)
        self.ui.Slider.valueChanged.connect(self.atualizarSlider)

        # IMPLEMENTAÇÃO DAS FUNÇÕES DO MENU
        # Menu Arquivo
        self.ui.actionAbrirImagem1.triggered.connect(self.abrirImagem1)
        self.ui.actionAbrirImagem2.triggered.connect(self.abrirImagem2)
        self.ui.actionSalvarResultado.triggered.connect(self.salvarImagem)
        self.ui.actionSair.triggered.connect(qApp.quit)
        # Menu Converte
        self.ui.actionTonsDeCinza.triggered.connect(self.converterParaCinzaMenu)
        self.ui.actionPretoBranco.triggered.connect(self.converterBinMenu)
        #Menu Operações
        #Operações Aritméticas
        self.ui.actionSoma.triggered.connect(self.somarMenu)
        self.ui.actionSubtracao.triggered.connect(self.subtrairMenu)
        self.ui.actionMultiplicacao.triggered.connect(self.multiplicarMenu)
        self.ui.actionDivisao.triggered.connect(self.dividirMenu)
        #Operações Lógicas
        self.ui.actionNOT.triggered.connect(self.fazerNotMenu)
        self.ui.actionAND.triggered.connect(self.fazerAndMenu)
        self.ui.actionOR.triggered.connect(self.fazerOrMenu)
        self.ui.actionXOR.triggered.connect(self.fazerXorMenu)
        #Transformações Geométricas
        self.ui.actionTranslacao.triggered.connect(self.transladarMenu)
        self.ui.actionRotacao.triggered.connect(self.rotacionarMenu)
        self.ui.actionEscalonamento.triggered.connect(self.escalonarMenu)
        #Deteccção de bordas
        self.ui.actionDerivativo1.triggered.connect(self.usarDerivativo1Menu)
        self.ui.actionDerivativo2.triggered.connect(self.usarDerivativo2Menu)
        self.ui.actionSobel.triggered.connect(self.usarSobelMenu)
        self.ui.actionKirsch.triggered.connect(self.usarKirschMenu)
        #Histograma
        self.ui.actionGerarHistograma.triggered.connect(self.gerarHistogramaMenu)
        self.ui.actionAutoescala.triggered.connect(self.fazerAutoescalaMenu)
        self.ui.actionEqualizacaoDeHistograma.triggered.connect(self.equalizarHistogramaMenu)
        self.ui.actionLimiarizacaoGlobal.triggered.connect(self.limiarizarGlobalMenu)
        self.ui.actionLimiarizacaoDeOtsu.triggered.connect(self.limiarizarOtsuMenu)
        #Filtro
        self.ui.actionFiltroDeMedia.triggered.connect(self.filtrarMediaMenu)
        self.ui.actionFiltroDeMediana.triggered.connect(self.filtrarMedianaMenu)
        self.ui.actionFiltroPassaAlta.triggered.connect(self.filtrarPassaAltaMenu)
        self.ui.actionLaplaciano.triggered.connect(self.filtrarLaplacianoMenu)
        #Morfologia
        self.ui.actionDilatacao.triggered.connect(self.dilatarMorfoMenu)
        self.ui.actionErosao.triggered.connect(self.erodirMorfoMenu)
        self.ui.actionAbertura.triggered.connect(self.abrirMorfoMenu)
        self.ui.actionFechamento.triggered.connect(self.fecharMorfoMenu)
        #Segmentação
        self.ui.actionSegmentacaoProposta.triggered.connect(self.segmentacaoPropostaMenu)
        #características
        self.ui.actionExtracaodeCaracteristicas.triggered.connect(self.extrairCaracteristicasMenu)
        #Projeto Final (OCR)
        self.ui.actionTransformadaDeHough.triggered.connect(self.transformadaDeHoughMenu)
        self.ui.actionCorrecaoDeOrientacao.triggered.connect(self.correcaoDeOrientacaoMenu)
        self.ui.actionOCR.triggered.connect(self.ocrMenu)


#############################################################################################################
####################                                                                     ####################
####################                         FUNÇÕES DE USO GERAL                        ####################
####################                                                                     ####################
#############################################################################################################

    def converterParaCinza(self,imagem,tipo):
        #Caso seja RGB
        if tipo == 1:
            imagem = imagem[:,:,0]*0.299 + imagem[:,:,1]*0.587 + imagem[:,:,2]*0.114 #conversão balanceada
        #Caso seja HSb
        elif tipo == 4:
            print("Não implementado")
        #Retorna o novo tipo
        tipo = 2

        imagem = imagem.astype('int')
        return (imagem,tipo)

    def converterBin(self,limiar,imagem,tipo):
        #Caso seja Mono ou Binário
        if tipo == 2 or tipo == 3:
            imagem[imagem > limiar] = 255
            imagem[imagem <= limiar] = 0
        #Caso contrário, não executar
        else:
            QMessageBox.warning(self, "Erro", "Imagem não é Mono ou binária")
        #Retornar o novo tipo
        tipo = 3

        return (imagem,tipo)

    def somarImagens(self,imagem1,imagem2):
        #soma todos os pixels de duas imagens de mesma dimensão
        imagem = imagem1 + imagem2

        return imagem

    def somarEscalar(self,imagem,escalar):
        #Soma todos o pixels de uma imagem com um escalar
        imagem = imagem + escalar

        imagem = imagem.astype('int')
        return imagem

    def subtrairImagens(self,imagem1,imagem2):
        #Subtrai todos os pixels de duas imagens de mesma dimensão
        imagem = imagem1 - imagem2

        return imagem

    def subtrairEscalar(self,imagem,escalar):
        #Subtrai todos o pixels de uma imagem com um escalar
        imagem = imagem + escalar

        imagem = imagem.astype('int')
        return imagem

    def multiplicarEscalar(self,imagem,escalar):
        #Multiplica todos o pixels de uma imagem com um escalar
        imagem = imagem*escalar

        imagem = imagem.astype('int')
        return imagem

    def dividirEscalar(self,imagem,escalar):
        #Divide todos o pixels de uma imagem com um escalar
        imagem = imagem/escalar

        imagem = imagem.astype('int')
        return imagem

    def escalonarImagem(self,img):
        #Escalona os valores dos pixels de uma imagem para ficarem entre 0 e 255
        #Se imagem tiver 3 canais
        if img.ndim == 3:
            for i in range(3):
                if (np.max(img[:,:,i]) - np.min(img[:,:,i])) != 0:
                    img[:,:,i] = (img[:,:,i] - np.min(img[:,:,i]))*(255/(np.max(img[:,:,i]) - np.min(img[:,:,i])))
        #Caso só possua 1 canal
        else:
            img = (img - np.min(img))*(255/(np.max(img) - np.min(img)))

        img = img.astype('int')
        return img

    def truncarImagem(self,imagem):
        #Trunca os valores dos pixels de uma imagem
        imagem[imagem < 0] = 0
        imagem[imagem > 255] = 255

        return imagem

    def fazerNot(self,imagem):
        #Converte 0 em 255 e 255 em 0
        imagem[imagem == 255] = -1
        imagem[imagem == 0] = 254
        imagem += 1

        return imagem

    def fazerAnd(self,imagem1,imagem2):
        #Fazer AND em todos os valores ao mesmo tempo
        imagem = np.logical_and(imagem1,imagem2)*255

        return imagem

    def fazerOr(self,imagem1,imagem2):
        #Fazer OR em todos os valores ao mesmo tempo
        imagem = np.logical_or(imagem1,imagem2)*255

        return imagem

    def fazerXor(self,imagem1,imagem2):
        #Fazer XOR em todos os valores ao mesmo tempo
        imagem = np.logical_or(np.logical_and(imagem1,np.logical_not(imagem2)),np.logical_and(np.logical_not(imagem1),imagem2))*255
        #imagem = self.fazerOr(self.fazerAnd(imagem1,self.fazerNot(imagem2)),self.fazerAnd(self.fazerNot(imagem1),imagem2))
        #imagem = np.logical_xor(imagem1,imagem2)*255

        return imagem

    def matrizTranslacao(self,dx,dy):
        #Construindo a matriz de translação
        mT = np.eye(3)
        mT[0,2] = dx
        mT[1,2] = dy
        #Preparanto mT para operações matriciais
        mT = np.matrix(mT)
        return mT

    def matrizRotacao(self,angulo):
        #Construindo a matriz de rotação
        mR = np.eye(3)
        anguloRad = angulo*np.pi/180
        mR[0,0] = np.cos(anguloRad)
        mR[0,1] = -np.sin(anguloRad)
        mR[1,0] = np.sin(anguloRad)
        mR[1,1] = np.cos(anguloRad)
        #Prepara mR para operações matriciais
        mR = np.matrix(mR)
        return mR

    def matrizEscalonamento(self,sx,sy):
        #Construindo a matriz de Escalonamento
        mE = np.eye(3)
        mE[0,0] = sx
        mE[1,1] = sy
        #Preparando mE para operações matricias
        mE = np.matrix(mE)
        return mE

    def transladar(self,imagem,tipo,dx,dy):
        #Descobre as dimensões da imagem
        linhas = imagem.shape[0]
        colunas = imagem.shape[1]
        #Constroi a matriz para translacao
        mT = self.matrizTranslacao(dx,dy)

        #Construindo matriz com todos os pontos
        pontos = np.array(np.meshgrid(np.arange(0,linhas),np.arange(0,colunas))).reshape(2,-1)
        pontos = np.vstack((pontos,np.ones([1,len(pontos[0,::])])))
        pontos = pontos.astype('int')
        pontos = np.matrix(pontos)

        #Calculando novas coordenadas
        novosPontos = mT*pontos
        novosPontos = novosPontos.astype('int')
        #Joga fora a linha de 1s
        novosPontos = novosPontos[0:2,::]
        pontos = pontos[0:2,::]

        #Caso Imagem seja RGB
        if tipo == 1:
            #Construindo nova imagem
            novaImagem = np.zeros([linhas,colunas,3])
            #Para cada canal
            for k in range(3):
                #Para cada ponto
                for i in range(linhas*colunas):
                    #Testa se algum ponto está fora
                    if novosPontos[0,i] >= 0 and novosPontos[0,i] < linhas and novosPontos[1,i] >= 0 and novosPontos[1,i] < colunas:
                        novaImagem[novosPontos[0,i],novosPontos[1,i],k] = imagem[pontos[0,i],pontos[1,i],k]
        #Caso imagem seja Mono ou binária
        elif tipo == 2 or tipo == 3:
            #Construindo nova imagem
            novaImagem = np.zeros([linhas,colunas])
            #Para cada ponto
            for i in range(linhas*colunas):
                #Testa se algum ponto está fora
                if novosPontos[0,i] >= 0 and novosPontos[0,i] < linhas and novosPontos[1,i] >= 0 and novosPontos[1,i] < colunas:
                    novaImagem[novosPontos[0,i],novosPontos[1,i]] = imagem[pontos[0,i],pontos[1,i]]
        #Caso imagem seja HSB
        elif tipo == 4:
            print("Não implementado")

        return novaImagem

    def rotacionar(self,imagem,tipo,angulo,coordX,coordY):
        #Descobre as dimensões da imagem
        linhas = imagem.shape[0]
        colunas = imagem.shape[1]
        #Constroi as matriz para rotacao em torno de um ponto
        mT1 = self.matrizTranslacao(-coordX,-coordY)
        mR = self.matrizRotacao(-angulo)
        mT2 = self.matrizTranslacao(coordX,coordY)
        mFinal = mT2*mR*mT1

        #Construindo matriz com todos os pontos
        pontos = np.array(np.meshgrid(np.arange(0,linhas),np.arange(0,colunas))).reshape(2,-1)
        pontos = np.vstack((pontos,np.ones([1,len(pontos[0,::])])))
        pontos = pontos.astype('int')
        pontos = np.matrix(pontos)

        #Calculando novas coordenadas
        novosPontos = mFinal*pontos
        novosPontos = novosPontos.astype('int')
        #Joga fora a linha de 1s
        novosPontos = novosPontos[0:2,::]
        pontos = pontos[0:2,::]

        #Caso Imagem seja RGB
        if tipo == 1:
            #Construindo nova imagem
            novaImagem = np.zeros([linhas,colunas,3])
            #Para cada canal
            for k in range(3):
                #Para cada ponto
                for i in range(linhas*colunas):
                    #Testa se algum ponto está fora
                    if novosPontos[0,i] >= 0 and novosPontos[0,i] < linhas and novosPontos[1,i] >= 0 and novosPontos[1,i] < colunas:
                        novaImagem[pontos[0,i],pontos[1,i],k] = imagem[novosPontos[0,i],novosPontos[1,i],k]
        #Caso imagem seja Mono ou binária
        elif tipo == 2 or tipo == 3:
            #Construindo nova imagem
            novaImagem = np.zeros([linhas,colunas])
            #Para cada ponto
            for i in range(linhas*colunas):
                #Testa se algum ponto está fora
                if novosPontos[0,i] >= 0 and novosPontos[0,i] < linhas and novosPontos[1,i] >= 0 and novosPontos[1,i] < colunas:
                    novaImagem[pontos[0,i],pontos[1,i]] = imagem[novosPontos[0,i],novosPontos[1,i]]
        #Caso imagem seja HSB
        elif tipo == 4:
            print("Não implementado")

        return novaImagem

    def escalonar(self,imagem,tipo,escalaX,escalaY,coordX,coordY):
        #Descobre as dimensões da imagem
        linhas = imagem.shape[0]
        colunas = imagem.shape[1]
        #Constroi as matriz para escalonamento a partir de um ponto
        mT1 = self.matrizTranslacao(-coordX,-coordY)
        mE = self.matrizEscalonamento(1/escalaX,1/escalaY)
        mT2 = self.matrizTranslacao(coordX,coordY)
        mFinal = mT2*mE*mT1

        #Construindo matriz com todos os pontos
        pontos = np.array(np.meshgrid(np.arange(0,linhas),np.arange(0,colunas))).reshape(2,-1)
        pontos = np.vstack((pontos,np.ones([1,len(pontos[0,::])])))
        pontos = pontos.astype('int')
        pontos = np.matrix(pontos)

        #Calculando novas coordenadas
        novosPontos = mFinal*pontos
        novosPontos = novosPontos.astype('int')
        #Joga fora a linha de 1s
        novosPontos = novosPontos[0:2,::]
        pontos = pontos[0:2,::]

        #Caso Imagem seja RGB
        if tipo == 1:
            #Construindo nova imagem
            novaImagem = np.zeros([linhas,colunas,3])
            #Para cada canal
            for k in range(3):
                #Para cada ponto
                for i in range(linhas*colunas):
                    #Testa se algum ponto está fora
                    if novosPontos[0,i] >= 0 and novosPontos[0,i] < linhas and novosPontos[1,i] >= 0 and novosPontos[1,i] < colunas:
                        novaImagem[pontos[0,i],pontos[1,i],k] = imagem[novosPontos[0,i],novosPontos[1,i],k]
        #Caso imagem seja Mono ou binária
        elif tipo == 2 or tipo == 3:
            #Construindo nova imagem
            novaImagem = np.zeros([linhas,colunas])
            #Para cada ponto
            for i in range(linhas*colunas):
                #Testa se algum ponto está fora
                if novosPontos[0,i] >= 0 and novosPontos[0,i] < linhas and novosPontos[1,i] >= 0 and novosPontos[1,i] < colunas:
                    novaImagem[pontos[0,i],pontos[1,i]] = imagem[novosPontos[0,i],novosPontos[1,i]]
        #Caso imagem seja HSB
        elif tipo == 4:
            print("Não implementado")

        return novaImagem

    def detectarDerivativo(self,imagem,tipoDerivativo):

        #Acha dimensões da imagem
        linhas, colunas = imagem.shape[0], imagem.shape[1]

        #Matrizes que vão receber os valores de d/dx e d/dy
        auxX = np.zeros([linhas,colunas])
        auxY = auxX;

        #Para o derivaivo 1
        if tipoDerivativo == 1:
            #Varrer ignorando contorno da imagem
            for i in range(1,linhas-1):
                for j in range(1,colunas-1):
                    auxX[i,j] = imagem[i,j] - imagem[i-1,j]
                    auxX[i,j] = imagem[i,j] - imagem[i,j-1]

        #Para o derivativo 2
        else:
            #Varrer ignorando contorno da imagem
            for i in range(1,linhas-1):
                for j in range(1,colunas-1):
                    auxX[i,j] = imagem[i+1,j] - imagem[i-1,j]
                    auxX[i,j] = imagem[i,j+1] - imagem[i,j-1]

        #Calculando intensidade e o sentido das bordas
        intensidadeBordas = np.sqrt(auxX**2 + auxY**2)
        SentidoBordas = np.arctan2(auxY,auxX)

        return (intensidadeBordas,SentidoBordas)

    def fazerConv2DTruncada(self,imagem,mascara):
        #Imagem não pode ser menor que máscara para essa função e funciona só para mascaras com N ímpar
        #ATUALMENTE EM DESUSO

        #Pegando as informações de linhas e colunas
        linhasIm, colunasIm = imagem.shape[0], imagem.shape[1]
        linhasMas, colunasMas = mascara.shape[0], mascara.shape[1]

        #Calcula o corte nos contornos da imagem resultante a depender do tamanho da máscara
        reduLinhas = int((linhasMas-1)/2)
        reduColunas = int((colunasMas-1)/2)

        #Define o resultado da convolução
        convolucao = np.zeros([linhasIm,colunasIm])
        #Calcula o resultado da convolução da máscara na imagem
        for i in range(reduLinhas,linhasIm-reduLinhas):
            for j in range(reduColunas,colunasIm-reduColunas):
                convolucao[i,j] = np.sum(mascara*imagem[i-reduLinhas:i+reduLinhas+1,j-reduColunas:j+reduColunas+1])

        #print(convolucao[1:-1,1:-1])
        return convolucao

    def convoluir2D(self,imagem,mascara,tipo):
        #Imagem não pode ser menor que máscara para essa função e funciona só para mascaras com N ímpar

        #Pegando as informações de linhas e colunas
        linhasIm, colunasIm = imagem.shape[0], imagem.shape[1]
        linhasMas, colunasMas = mascara.shape[0], mascara.shape[1]

        #Calcula o corte nos contornos da imagem resultante a depender do tamanho da máscara
        reduLinhas = linhasMas-1
        reduColunas = colunasMas-1

        #Define o resultado da convolução
        convolucao = np.zeros([linhasIm+linhasMas-1,colunasIm+colunasMas-1])
        #Calcula o resultado da convolução da máscara na imagem
        for i in range(linhasMas):
            for j in range(colunasMas):
                convolucao[i:i+linhasIm,j:j+colunasIm] = convolucao[i:i+linhasIm,j:j+colunasIm] + mascara[linhasMas-1-i,colunasMas-1-j]*imagem

        #print(convolucao[reduLinhas:-reduLinhas,reduColunas:-reduColunas])
        #tipo 1: quer convolucao completa
        if tipo == 1:
            return convolucao
        #tipo 2: quer somente resultado com superposição completa entre matriz e máscara
        elif tipo == 2:
            return convolucao[reduLinhas:-reduLinhas,reduColunas:-reduColunas]
        #tipo 3: quer resultado com mesma ou quase mesma dimensão da imagem original
        elif tipo == 3:
            if linhasMas%2!=0:
                #auxLinSup = reduLinhas-1
                #auxLinInf = reduLinhas-1
                auxLinSup = int(reduLinhas/2)
                auxLinInf = int(reduLinhas/2)
            else:
                #auxLinSup = reduLinhas-1
                #auxLinInf = reduLinhas-2
                auxLinSup = int((reduLinhas+1)/2)
                auxLinInf = int((reduLinhas-1)/2)

            if colunasMas%2!=0:
                #auxColEsq = reduColunas-1
                #auxColDir = reduColunas-1
                auxColEsq = int(reduColunas/2)
                auxColDir = int(reduColunas/2)
            else:
                #auxColEsq = reduColunas-1
                #auxColDir = reduColunas-2
                auxColEsq = int((reduColunas+1)/2)
                auxColDir = int((reduColunas-1)/2)
            #Caso o número de linhas e colunas da máscara seja ímpar, retorna imagem com mesma dimensão da original
            return convolucao[auxLinSup:-auxLinInf,auxColEsq:-auxColDir]

    def detectarSobel(self, imagem):
        #Acha dimensões da imagem
        linhas, colunas = imagem.shape[0], imagem.shape[1]
        #limiar predefinido
        limiar = 51

        #primeira matriz é Sx e a segunda é Sy
        ms = np.zeros([3,3,2])
        #Sx
        ms[::,::,0] = np.array([[-1,0,1],
                                 [-2,0,2],
                                 [-1,0,1]])
        #Sy
        ms[::,::,1] = ms[::,::,0].T
        #Fator de correção
        escala = 1/4

        aux = np.zeros([linhas,colunas,2])
        for i in range(2):
            aux[::,::,i] = self.convoluir2D(imagem,ms[::,::,i],3)
        aux = aux*escala

        #Calculando intensidade e o sentido das bordas
        intensidadeBordas = np.abs(aux[::,::,0]) + np.abs(aux[::,::,1])
        SentidoBordas = np.arctan2(aux[::,::,1],aux[::,::,0])

        #intensidadeBordas = self.escalonarImagem(intensidadeBordas)
        return (intensidadeBordas,SentidoBordas)

    def detectarKirsch(self, imagem):
        #Acha dimensões da imagem
        linhas, colunas = imagem.shape[0], imagem.shape[1]

        #Definindo as 8 máscaras de Kirsch
        K = np.zeros([3,3,8])
        K[::,::,0] = np.array([[-3,-3,5],
                               [-3, 0,5],
                               [-3,-3,5]])
        K[::,::,1] = np.array([[-3, 5, 5],
                               [-3, 0, 5],
                               [-3,-3,-3]])
        K[::,::,2] = np.array([[ 5, 5, 5],
                               [-3, 0,-3],
                               [-3,-3,-3]])
        K[::,::,3] = K[::,::-1,1]
        K[::,::,4] = K[::,::-1,0]
        K[::,::,5] = np.array([[-3,-3,-3],
                               [ 5, 0,-3],
                               [ 5, 5,-3]])
        K[::,::,6] = K[::-1,::,2]
        K[::,::,7] = K[::,::-1,5]
        #Fator de correção
        escala = 1/15

        #Calcula as 8 convoluções
        aux = np.zeros([linhas,colunas,8])
        for i in range(8):
            #aux[::,::,i] = self.fazerConv2DTruncada(imagem,K[::,::,i])
            aux[::,::,i] = self.convoluir2D(imagem,K[::,::,i],3)
        aux = aux*escala

        #Inicializa as variáveis que vão receber os valores de intensidade e sentido
        intensidadeBordas = np.zeros([linhas,colunas])
        SentidoBordas = np.zeros([linhas,colunas])
        #Testa qual Kernel é o mais intenso, armazena e calcula o sentido do gradiente
        for i in range(linhas):
            for j in range(colunas):
                intensidadeBordas[i,j] = np.max(np.abs(aux[i,j,::]))
                indice = np.argmax(np.abs(aux[i,j,::]))
                SentidoBordas[i,j] = indice*np.pi/4

        #intensidadeBordas = self.escalonarImagem(intensidadeBordas)
        return (intensidadeBordas,SentidoBordas)

    def histograma(self,imagem):

        P = np.zeros(256)
        for i in range(256):
            P[i] = imagem[imagem == i].size
        P = P/imagem.size

        return P

    def plotarHistograma(self,P,titulo):

        plt.close(titulo)
        fig = plt.figure(num = titulo, clear = True)
        # ax = plt.subplots(num=1, clear=True)#, num = titulo)
        ax = plt.axes()
        ax.bar(np.arange(0,256),P)
        ax.set_title("Histograma")
        ax.set_xlabel("pixels")
        ax.set_ylabel("Probabilidade")
        #plt.tight_layout()
        fig.show()

    def equalizarHistograma(self,histograma,imagem):
        #Acha dimensões da imagem
        linhas, colunas = imagem.shape[0], imagem.shape[1]

        #Calcula a função cumulativa
        cdf = np.cumsum(histograma)
        inter = np.arange(0,256)/255

        #Inicializa matriz que recebe os valores
        aux = np.zeros([linhas,colunas])
        for i in range(256):
            #Retorna o índice (pixel) em que o valor de cdf se assemelha ao máximo com o valor da função alvo
            indice = np.argmin(np.abs(cdf[i]-inter))
            aux[imagem == i] = indice

        return cdf, aux

    def plotarCumulativa(self, cdf,histograma):
        aux = np.cumsum(histograma)

        plt.close('CDF')
        fig = plt.figure(num = 'CDF', clear = True)

        ax = plt.axes()
        ax.plot(np.arange(0,256),cdf,label='cdf anterior')
        ax.plot(np.arange(0,256),aux,label='cdf nova')
        #ax.plot(np.arange(0,256),np.arange(0,256)/255,label='cdf certa')
        ax.legend()
        ax.set_title("CDF")
        ax.set_xlabel("pixels")
        ax.set_ylabel("Probabilidade")
        plt.grid()
        #plt.tight_layout()
        fig.show()

    def limiarizarIterativo(self,histograma,imagem):

        contagem = histograma*imagem.size

        #Media dos pixels de uma imagem para inicialização de limiar
        t = sum(histograma*np.arange(0,256))
        dt = 2 #Inicialização para o dT
        while(dt > 1):
            #Definindo limites dos gruspos
            fim1 = int(np.floor(t))
            inicio2 = fim1 + 1

            #Separação dos grupos
            grupo1 = contagem[0:fim1+1]/sum(contagem[0:fim1+1])
            grupo2 = contagem[inicio2:256]/sum(contagem[inicio2:256])

            #Calculando médias dos grupos
            media1 = sum(grupo1*np.arange(0,fim1+1))
            media2 = sum(grupo2*np.arange(inicio2,256))

            #Atualizando variáveis
            novoT = (media1 + media2)/2
            dt = abs(novoT - t)
            if dt > 1:
                t = novoT

        t = int(round(t))
        return t

    def limiarizarOtsu(self,histograma,imagem):

        cdf = np.cumsum(histograma)
        media = sum(histograma*np.arange(0,256))
        #media = np.mean(imagem)
        var = sum(((np.arange(0,256) - media)**2)*histograma)
        #print('media = ',media,' ',np.mean(imagem),'  ','variancia = ',var,' ',np.var(imagem))
        limiar = 0
        maiorVar = 0
        for k in range(1,255):

            P1 = sum(histograma[0:k+1])
            #P2 = sum(histograma[k+1:256])
            P1 = cdf[k]
            m1 = sum(np.arange(0,k+1)*histograma[0:k+1])
            #m2 = sum(np.arange(k+1,256)*histograma[k+1:256])
            #print((media*P1 - m),P1,(P1*(1-P1)))

            #varB1 = P1*(m1 - media)**2 + P2*(m2 - media)**2

            if (P1*(1-P1)) != 0:
                varB = ((media*P1 - m1)**2)/(P1*(1-P1))
            else:
                varB = 0

            #print(k, P1, varB1, varB2)


            if varB > maiorVar:
                maiorVar = varB
                limiar = k
                #print(k, maiorVar)

        return limiar

    def filtrarMedia(self,imagem,tipo,mLinhas,mColunas):
        #Acha dimensões da imagem
        linhas, colunas = imagem.shape[0], imagem.shape[1]

        #Montando máscara
        m = np.ones([mLinhas,mColunas])

        #Caso imagem seja colorida
        if tipo == 1:
            resultante = np.zeros([linhas,colunas,3])
            for i in range(3):
                resultante[::,::,i] = self.convoluir2D(imagem[::,::,i],m,3)
        #Caso seja mono ou binária
        elif tipo == 2 or tipo == 3:
            resultante = self.convoluir2D(imagem,m,3)

        #dividindo pelo fator de correção
        resultante = resultante/(mLinhas*mColunas)
        resultante = resultante.astype('int')

        return resultante

    def filtrarMediana(self, imagem,tipo):
        #Acha dimensões da imagem
        linhas, colunas = imagem.shape[0], imagem.shape[1]

        #Caso imagem seja colorida
        if tipo == 1:
            resultante = np.zeros([linhas-2,colunas-2,3])
            #Para cada canal
            for k in range(3):
                #Calculo da mediana para cada pixel
                for i in range(1,linhas-2):
                    for j in range(1,colunas-2):
                        resultante[i,j,k] = np.median(imagem[i-1:i+2,j-1:j+2,k])
        #Caso seja mono ou binária
        elif tipo == 2 or tipo == 3:
            resultante = np.zeros([linhas-2,colunas-2])
            #Calculo da mediana para cada pixel
            for i in range(1,linhas-2):
                for j in range(1,colunas-2):
                    resultante[i,j] = np.median(imagem[i-1:i+2,j-1:j+2])
        return resultante

    def filtrarPassaAlta(self,imagem,tipo):
        #Acha dimensões da imagem
        linhas, colunas = imagem.shape[0], imagem.shape[1]

        #Montando máscara
        m = np.ones([3,3])*(-1)
        m[1,1] = 8

        #Caso imagem seja colorida
        if tipo == 1:
            resultante = np.zeros([linhas,colunas,3])
            for i in range(3):
                resultante[::,::,i] = self.convoluir2D(imagem[::,::,i],m,3)
        #Caso seja mono ou binária
        elif tipo == 2 or tipo == 3:
            resultante = self.convoluir2D(imagem,m,3)

        #dividindo pelo fator de correção
        #resultante = resultante/8
        resultante = self.truncarImagem(resultante)
        resultante = resultante.astype('int')

        return resultante

    def filtrarLaplaciano(self,imagem,tipo):
        #Acha dimensões da imagem
        linhas, colunas = imagem.shape[0], imagem.shape[1]

        #Montando máscara
        m = np.zeros([3,3])
        m[0,1] = -1
        m[2,1] = -1
        m[1,0] = -1
        m[1,2] = -1
        m[1,1] = 4

        #Caso imagem seja colorida
        if tipo == 1:
            resultante = np.zeros([linhas,colunas,3])
            for i in range(3):
                resultante[::,::,i] = self.convoluir2D(imagem[::,::,i],m,3)
        #Caso seja mono ou binária
        elif tipo == 2 or tipo == 3:
            resultante = self.convoluir2D(imagem,m,3)
        #dividindo pelo fator de correção
        #resultante = resultante/4
        resultante = self.truncarImagem(resultante)
        resultante = resultante.astype('int')

        return resultante

    def fazerDilatacao(self,imagem,elemento,cor):
        #Achar dimensão do elemento estruturante
        dim = elemento.shape[0]
        #Acha dimensões da imagem
        linhas, colunas = imagem.shape[0], imagem.shape[1]

        #Caso seja deseje ser setado os elementos pretos em vez de brancos
        if cor == 1:
            imagem = self.fazerNot(imagem)

        #Índices para translações
        ind = np.where(elemento == 1)
        ind = np.array([ind[0],ind[1]])

        #Dimensão da imagem resultante é: linhas + (dim-1)/2 X colunas + (dim-1)/2
        resultante = np.zeros([linhas+int(dim-1),colunas+int(dim-1)])

        #Somatório lógico das translações
        for i in range(len(ind[0,::])):
            resultante[ind[0,i]:(ind[0,i]+linhas),ind[1,i]:(ind[1,i]+colunas)] = np.logical_or(resultante[ind[0,i]:(ind[0,i]+linhas),ind[1,i]:(ind[1,i]+colunas)],imagem)*255
        resultante = resultante.astype('int')

        #voltar a imagem para a cor original
        if cor == 1:
            resultante = self.fazerNot(resultante)

        return resultante

    def fazerErosao(self,imagem,elemento,cor):
        #Achar dimensão do elemento estruturante
        dim = elemento.shape[0]
        #Acha dimensões da imagem
        linhas, colunas = imagem.shape[0], imagem.shape[1]
        #Aumento nas dimensões da imagem nas bordas
        aumento = int((dim-1)/2)

        #Caso seja deseje ser setado os elementos pretos em vez de brancos
        if cor == 1:
            imagem = self.fazerNot(imagem)

        #Índices para testes
        ind = np.where(elemento == 1)
        ind = np.array([ind[0],ind[1]])
        ind = ind - int((dim-1)/2) #deslocamentos

        #Imagem na qual serão feitos os testes
        aux = np.zeros([linhas+2*aumento,colunas+2*aumento])
        aux[aumento:-aumento,aumento:-aumento] = imagem #Acréscimo de bordas zeradas
        aux = (aux/255).astype('int')

        resultante = np.zeros([linhas+2*aumento,colunas+2*aumento])
        for i in range(aumento,linhas+aumento*2-aumento):
            for j in range(aumento,colunas+aumento*2-aumento):
                #Testando todos os pixels setados
                auxTeste = elemento*aux[i-aumento:i+aumento+1,j-aumento:j+aumento+1]
                #Se tiver alguma diferença, não setar pixel
                if np.array_equal(auxTeste,elemento):
                    resultante[i,j] = 255

        #Recortando o resultado para ter a mesma dimensão da imagem original
        resultante = resultante[aumento:-aumento,aumento:-aumento]
        resultante = resultante.astype('int')

        #voltar a imagem para a cor original
        if cor == 1:
            resultante = self.fazerNot(resultante)

        return resultante

    def fazerAbertura(self,imagem,elemento,cor):
        #Achar dimensão do elemento estruturante
        dim = elemento.shape[0]
        #Aumento nas dimensões da imagem nas bordas
        aumento = int((dim-1)/2)

        #Executando erosão com o espelho do elemento e em seguida a dilatação
        resultadoTemp = self.fazerErosao(imagem,np.flip(elemento),cor)
        resultado = self.fazerDilatacao(resultadoTemp,elemento,cor)

        #Recortando o resultado para ter a mesma dimensão da imagem original
        resultado = resultado[aumento:-aumento,aumento:-aumento]

        return resultado

    def fazerFechamento(self,imagem,elemento,cor):
        #Achar dimensão do elemento estruturante
        dim = elemento.shape[0]
        #Aumento nas dimensões da imagem nas bordas
        aumento = int((dim-1)/2)

        #Executando a dilatação com o espelho do elemento e em seguida a Erosão
        resultadoTemp = self.fazerDilatacao(imagem,np.flip(elemento),cor)
        resultado = self.fazerErosao(resultadoTemp,elemento,cor)

        #Recortando o resultado para ter a mesma dimensão da imagem original
        resultado = resultado[aumento:-aumento,aumento:-aumento]

        return resultado

    def segmentacaoProposta(self,img,limiar):
        #Acha dimensões da imagem
        lin, col = img.shape[0], img.shape[1]

        #Definindo uma imagem com bordas zeradas
        imagem = np.zeros([lin+2,col+2])
        imagem[1:-1,1:-1] = img/255

        linhas, colunas = imagem.shape[0], imagem.shape[1]
        #Posições para comparação
        testeCE = np.array([[-1,-1,-1, 0],
                            [-1, 0, 1,-1]])
        testeDA = np.array([[ 0, 1, 1, 1],
                            [ 1,-1, 0, 1]])
        teste = np.array([[ 0,-1,-1,-1, 0, 0, 1, 1, 1],
                          [ 0,-1, 0, 1,-1, 1,-1, 0, 1]])

        #Inicializando o mapa de regiões e o contador de regiões
        mapa = np.zeros([linhas,colunas])
        regioes = 0

        #Varredura da imagem
        for i in range(1,linhas-1):
            for j in range(1,colunas-1):
                #Se for pixel ativo
                if imagem[i,j] != 0:
                    #Caso não haja nenhuma região ainda
                    if regioes == 0:
                        regioes = 1
                        mapa[i,j] = 1
                        #mapa[i+testeDA[0,::],j+testeDA[1,::]] = 1*imagem[i+testeDA[0,::],j+testeDA[1,::]]
                    else:
                        #Testa se pixels ativos de cima e da esquerda possuem região
                        aux = np.trim_zeros(np.unique(mapa[i+testeCE[0,::],j+testeCE[1,::]]*imagem[i+testeCE[0,::],j+testeCE[1,::]]))
                        #Caso não possuam, definir nova região
                        if len(aux) == 0:
                            regioes = regioes + 1
                            mapa[i,j] = regioes
                            #mapa[i+testeDA[0,::],j+testeDA[1,::]] = regioes*imagem[i+testeDA[0,::],j+testeDA[1,::]]
                        #Caso possuam, setar para a menor região caso haja mais de uma
                        else:
                            mapa[i,j] = aux[0]
                            #mapa[i+testeDA[0,::],j+testeDA[1,::]] = mapa[i,j]*imagem[i+testeDA[0,::],j+testeDA[1,::]]

        #Combinando regiões adjacentes
        for i in range(1,linhas-1):
            for j in range(1,colunas-1):
                #Se for pixel ativo
                if mapa[i,j] != 0:
                    aux = np.trim_zeros(np.unique(mapa[i+teste[0,::],j+teste[1,::]]*imagem[i+teste[0,::],j+teste[1,::]]))
                    #Caso haja outra região ao redor do pixel, setar todos os pixels da região atual para a de menor valor
                    if len(aux) > 1:
                        mapa[mapa == mapa[i,j]] = aux[0]

        #Verifica a quantidade de pixels de cada região em relação a imagem toda
        lenRegioes = np.zeros(regioes)
        for i in range(regioes):
            lenRegioes[i] = len(mapa[mapa == i+1])/(lin*col)

        #Se alguma região tiver quantidade relativa de pixels menor que o limiar, ela é excluida
        for i in range(regioes):
            if lenRegioes[i] < limiar:
                mapa[mapa == i+1] = 0

        #Verificando quais regiões ainda existem depois dos dois processos
        reg = np.trim_zeros(np.unique(mapa))
        #Definindo matriz temporária que vai receber os novos rótulos de região
        auxM = np.zeros([linhas,colunas])
        cont = 1
        for i in reg:
            auxM[mapa == i] = cont
            cont = cont + 1

        #Atualizado novo mapa com os valores corrigidos
        mapa = auxM
        #Atualizando nova quantidade de regiões
        regioes = len(reg)

        #print(mapa[1:-1,1:-1])
        #print(regioes)

        mapa = mapa.astype('int')
        return (mapa[1:-1,1:-1],regioes)

    def extrairCaracteristicas(self,regiao):
        #Faz uma cópia do mapa da região e calcula total de pixels da imagem
        mapa = regiao.mapa.copy()
        total = mapa.size

        #Todas as posições de pixels ativos
        ind = np.where(mapa == 1)
        ind = np.array([ind[0],ind[1]]) #ind[0] linhas, ind[1] colunas

        #Calcula área
        area = len(ind[0,::])

        #Calcula centros de massas
        centroDeMassa = []
        centroDeMassa.append(sum(ind[1,::])/area) #Eixo X
        centroDeMassa.append(sum(ind[0,::])/area) #Eixo Y

        #Calcula translação pelos centros de massa
        des = np.array([[centroDeMassa[1]],[centroDeMassa[0]]])
        indTrans = ind - des

        #Calcula parâmetros a, b e c para calcular o ângulo de rotação que minimize o momento de inércia
        a = sum(indTrans[1,::]**2) #x'^2
        b = 2*sum(indTrans[1,::]*indTrans[0,::]) #x'*y'
        c = sum(indTrans[0,::]**2) #y'^2
        #Calculando o ângulo de rotação a partir dos parâmetros a,b e c
        if (b == 0 and a == c):
            thetaRad = 0
            theta = 0
        else:
            thetaRad = np.arctan2(b,(a-c))/2
            theta = thetaRad*180/np.pi


        #Definindo uma matriz de rotações assumindo como entrada vetor [y x]'
        mR = np.matrix([[ np.cos(thetaRad),-np.sin(thetaRad)],
                        [ np.sin(thetaRad), np.cos(thetaRad)]])
        #Aplicando rotação aos vetores de posição
        indRot = mR*indTrans

        #Calculando o comprimento a partir do eixo X
        maximo = np.max(indRot[1,::])
        minimo = np.min(indRot[1,::])
        comprimento = maximo - minimo #Eixo X
        #Calculando a largura a partir do eixo Y
        largura =  np.max(indRot[0,::]) - np.min(indRot[0,::]) #Eixo Y

        #Criando vetor de posições dos extremos orientação ideal e mudando de volta para a orientação e posição originais
        extremos = [[     0,     0],
                    [minimo,maximo]]
        extremos = np.array(extremos)
        extremos = mR.T*extremos
        extremos = extremos + des

        #Alterando atributos da região para os calculados
        regiao.area = area
        regiao.centroDeMassa = centroDeMassa
        regiao.orientacao = -theta
        regiao.comprimento = comprimento
        regiao.largura = largura
        regiao.extremos = extremos

    def transformadaDeHough(self,imagem):
        linhas, colunas = imagem.shape[0], imagem.shape[1]

        distancia = np.floor(np.sqrt(linhas**2 + colunas**2))
        theta = np.arange(-90,90.2,0.2)
        p = np.arange(-distancia,distancia+1,1)

        matriz = np.zeros([p.size,theta.size])

        ind = np.where(imagem != 255)
        ind = np.array([ind[0],ind[1]]) #ind[0] linhas, ind[1] colunas
        x = ind[0,::]
        y = ind[1,::]

        #for i in range(x.size):
        #    for j in range(y.size):
                #for k in range(theta.size):
        #        auxP = np.round(x[i]*np.cos(theta*np.pi/180)+y[j]*np.sin(theta*np.pi/180))

        for i in range(theta.size):
            auxP = np.round(x*np.cos(theta[i]*np.pi/180)+y*np.sin(theta[i]*np.pi/180))
            #for j in range(x.size):
            #    matriz[i,np.where(auxP[j] == p)] = matriz[i,np.where(auxP[j] == p)] + 1
            for j in range(p.size):
                matriz[j,i] += np.where(auxP == p[j])[0].size
            #for j in range(auxP.size):
            #    matriz[np.where(p == auxP[j]),i] += 1

        i = np.where(matriz == np.max(matriz))
        #print(theta[i[1]])

        maximo = np.max(matriz)

        auxMatriz = matriz.copy()
        auxMatriz[np.where(auxMatriz < 0.9*maximo)] = 0

        angulo = theta[np.argmax(np.sum(auxMatriz,0))]
        print(angulo)

        return matriz,angulo

    def novoRotacionar(self,imagem,tipo,angulo):
        #Descobre as dimensões da imagem
        linhas = imagem.shape[0]
        colunas = imagem.shape[1]
        #Centro da imagem
        coordX = int(linhas/2)
        coordY = int(colunas/2)

        #Constroi as matriz para rotacao em torno de um ponto
        mT1 = self.matrizTranslacao(-coordX,-coordY)
        mR = self.matrizRotacao(-angulo)
        mT2 = self.matrizTranslacao(coordX,coordY)
        mFinal = mT2*mR*mT1

        #Construindo matriz com todos os pontos
        pontos = np.array(np.meshgrid(np.arange(0,linhas),np.arange(0,colunas))).reshape(2,-1)
        pontos = np.vstack((pontos,np.ones([1,len(pontos[0,::])])))
        pontos = pontos.astype('int')
        pontos = np.matrix(pontos)

        #Calculando novas coordenadas
        novosPontos = mFinal*pontos
        novosPontos = novosPontos.astype('int')
        #Joga fora a linha de 1s
        novosPontos = novosPontos[0:2,::]
        pontos = pontos[0:2,::]

        #Caso Imagem seja RGB
        if tipo == 1:
            #Construindo nova imagem
            novaImagem = np.ones([linhas,colunas,3])*255
            #Para cada canal
            for k in range(3):
                #Para cada ponto
                for i in range(linhas*colunas):
                    #Testa se algum ponto está fora
                    if novosPontos[0,i] >= 0 and novosPontos[0,i] < linhas and novosPontos[1,i] >= 0 and novosPontos[1,i] < colunas:
                        novaImagem[pontos[0,i],pontos[1,i],k] = imagem[novosPontos[0,i],novosPontos[1,i],k]
        #Caso imagem seja Mono ou binária
        elif tipo == 2 or tipo == 3:
            #Construindo nova imagem
            novaImagem = np.ones([linhas,colunas])*255
            #Para cada ponto
            for i in range(linhas*colunas):
                #Testa se algum ponto está fora
                if novosPontos[0,i] >= 0 and novosPontos[0,i] < linhas and novosPontos[1,i] >= 0 and novosPontos[1,i] < colunas:
                    novaImagem[pontos[0,i],pontos[1,i]] = imagem[novosPontos[0,i],novosPontos[1,i]]
        #Caso imagem seja HSB
        elif tipo == 4:
            print("Não implementado")

        return novaImagem

    def segmentacaoLetras(self,imagem):
        lin, col = imagem.shape[0], imagem.shape[1]

        ####################################################################################################################
        ################################################# DEFININDO LINHAS #################################################
        ####################################################################################################################

        #Matriz que vai armazenar limites superior e inferior das linhas do texto
        linhas = np.matrix([0,0],dtype='int')
        #Indicador se está buscando linha superior ou inferior (0 para linha superior e 1 para linha inferior)
        flag = 0
        #Indicador de qual linha está sendo buscada
        cont = 0
        for i in range(lin):
            #Indicador se uma linha é completamente branca
            testeBranca = 0
            for j in range(col):
                #Caso esteja buscando o limite superior da linha
                if flag == 0:
                    #Caso encontre algum pixel preto, encontra parte superior da linha
                    if imagem[i,j] == 0:
                        flag = 1
                        #gravando limite superior da linha
                        linhas[cont,0] = i-1
                        #Impede de usar a mesma linha para limite inferior
                        testeBranca += 1
                #Caso ele esteja buscando o limite inferior da linha
                else:
                    #Caso haja um pixel preto, teste é incrementado e não passa na condição
                    if imagem[i,j] == 0:
                        testeBranca += 1

            #Caso esteja buscando o limite inferior da linha
            if flag == 1:
                #Caso todos os pixels da linha tenham sido brancos
                if testeBranca == 0:
                    flag = 0
                    #gravando limite inferior da linha
                    linhas[cont,1] = i
                    #Atualiza contador
                    cont += 1
                    linhas = np.vstack((linhas,np.matrix([0,0])))

        #Excluindo informações incompletas
        if linhas[cont,1] == 0:
            #Deleta a última linha da matriz
            linhas = np.delete(linhas,cont,axis=0)

        #salvando número de linhas do texto
        nLinhas = linhas.shape[0]

        ####################################################################################################################
        ################################################ DEFININDO COLUNAS #################################################
        ####################################################################################################################

        #Criação de matriz que vai carregar informação de todos os limites dos caracteres
        limites = np.matrix([0,0,0,0,0],dtype='int')
        #Indicador se está buscando coluna esquerda ou direita (0 para coluna esquerda e 1 para coluna direita)
        flag = 0
        #Indicador de caracter
        cont = 0
        #Contador de letras em cada linha do texto
        letras = np.array([0],dtype='int')
        #Para cada linha do texto
        for k in range(nLinhas):
            #Varredura de cada coluna
            for j in range(col):
                #Indicador se uma coluna é completamente branca
                testeBranca = 0

                #Varredura das linhas a partir das linhas do texto
                for i in range(linhas[k,1]-linhas[k,0]+1):
                    #Caso esteja buscando o limite a esquerda
                    if flag == 0:
                        #Caso tenha encontrago algum pixel preto
                        if imagem[linhas[k,0]+i,j] == 0:
                            flag = 1
                            #Grava limites superior e inferior
                            limites[cont,0] = linhas[k,0]
                            limites[cont,1] = linhas[k,1]
                            #Grava limite esquerdo do caracter
                            limites[cont,2] = j-1
                            #Impede de usar a mesma linha para limite inferior
                            testeBranca += 1
                    #Caso esteja bvuscando o limite a direita
                    else:
                        #Caso haja um pixel preto, teste é incrementado e não passa na condição
                        if imagem[linhas[k,0]+i,j] == 0:
                            testeBranca += 1

                if flag == 1:
                    #Caso todos os pixels da coluna tenham sido brancos
                    if testeBranca == 0:
                        flag = 0
                        #gravando limite direito do caracter
                        limites[cont,3] = j
                        limites[cont,4] = k + 1
                        #Atualiza contadores
                        letras[k] += 1
                        cont += 1

                        limites = np.vstack((limites,np.matrix([0,0,0,0,0])))

            letras = np.append(letras,[0])
            limites = limites.astype('int')
        #Excluindo informações incompletas
        if limites[cont,3] == 0:
            #Deleta a última linha da matriz
            limites = np.delete(limites,cont,axis=0)
        #Deletar o último item vazio
        letras = np.delete(letras,nLinhas)

        ####################################################################################################################
        ############################################# AGLUTINAÇÃO DE PALAVRAS ##############################################
        ####################################################################################################################

        #Adiciona uma nova coluna para delimitação de palavras/caracteres unidos
        limites = np.hstack((limites,np.zeros([limites.shape[0],1])))

        pos = 0
        regiao = 1
        for i in range(nLinhas):

            posEsquerda = limites[pos:pos+letras[i],2]
            posDireita = limites[pos:pos+letras[i],3]

            derivada = posEsquerda[1::] - posDireita[0:-1:]
            limiar = 0.5*np.max(derivada)


            #fig = plt.figure()
            #ax = plt.axes()
            #ax.plot(np.arange(1,derivada.shape[0]+1),derivada)
            #fig.show()

            for j in range(letras[i]):
                #Primeiro caracter da linha não passa por condição
                if j == 0:
                    limites[pos+j,5] = regiao
                #Demais caracteres da linha
                else:
                    #Caso valor da derivada passe de um determinado limiar, mudar região
                    if derivada[j-1] >= limiar:
                        regiao += 1

                    limites[pos+j,5] = regiao

            pos += letras[i]
            regiao += 1

        limites = limites.astype('int')

        ####################################################################################################################
        ############################################# RECORTE E PADRONIZAÇÃO ###############################################
        ####################################################################################################################

        letrasDicionario = {}
        for i in range(limites.shape[0]):
            #Recorte do caracter na imagem original
            supe = limites[i,0]
            infe = limites[i,1]
            esqu = limites[i,2]
            dire = limites[i,3]
            letrasDicionario[i] = imagem[supe+1:infe,esqu+1:dire].copy()

            #Dimensões do recorte do caracter
            m , n = letrasDicionario[i].shape[0], letrasDicionario[i].shape[1]
            #Testa se a matriz e quadrada e completa com 255 a parte que não é
            dif = m - n
            if dif > 0:
                if dif%2 == 0:
                    letrasDicionario[i] = np.hstack((255*np.ones([m,int(dif/2)]),letrasDicionario[i],255*np.ones([m,int(dif/2)])))
                else:
                    letrasDicionario[i] = np.hstack((255*np.ones([m,int((dif-1)/2)]),letrasDicionario[i],255*np.ones([m,int((dif-1)/2)])))
                    letrasDicionario[i] = np.hstack((letrasDicionario[i],255*np.ones([m,1])))
                if m%2 == 0:
                    letrasDicionario[i] = np.hstack((letrasDicionario[i],255*np.ones([m,1])))
                    letrasDicionario[i] = np.vstack((letrasDicionario[i],255*np.ones([1,m+1])))
            elif dif < 0:
                if dif%2 == 0:
                    letrasDicionario[i] = np.vstack((255*np.ones([int(-dif/2),n]),letrasDicionario[i],255*np.ones([int(-dif/2),n])))
                else:
                    letrasDicionario[i] = np.vstack((255*np.ones([int((-dif-1)/2),n]),letrasDicionario[i],255*np.ones([int((-dif-1)/2),n])))
                    letrasDicionario[i] = np.vstack((letrasDicionario[i],255*np.ones([1,n])))
                if n%2 == 0:
                    letrasDicionario[i] = np.vstack((letrasDicionario[i],255*np.ones([1,n])))
                    letrasDicionario[i] = np.hstack((letrasDicionario[i],255*np.ones([n+1,1])))
            else:
                if n%2 == 0:
                    letrasDicionario[i] = np.vstack((letrasDicionario[i],255*np.ones([1,n])))
                    letrasDicionario[i] = np.hstack((letrasDicionario[i],255*np.ones([n+1,1])))
            #print(letrasDicionario[i].shape)

        flag = 1
        for i in range(limites.shape[0]):
            nome = "testeLetra" + str(i+1)
            self.salvarImagens(letrasDicionario[i],nome,flag)
            flag = 0

        self.salvarLinhasRegioes(limites)

        #self.desenharCaixas(limites)

        print(nLinhas,letras)
        return limites

    def salvarImagens(self,imagem,nome,flag):
        #Caso não haja diretório, criar um
        if not os.path.exists("imagensLetras"):
            os.mkdir("imagensLetras")
        #caso haja diretório
        else:
            #Caso seja a primeira chamada da função
            if flag == 1:
                shutil.rmtree("imagensLetras")
                os.mkdir("imagensLetras")
        path = "imagensLetras//"

        #Preenchendo os 3 canais
        aux = np.zeros(imagem.shape + tuple([3]))
        for i in range(3):
            aux[:,:,i] = imagem
        #Salvando arquivo
        cv2.imwrite(os.path.join(path,nome + ".png"),aux)

    def salvarLinhasRegioes(self,limites):
        path = "imagensLetras//"
        np.savetxt(os.path.join(path,"linhasRegioes.txt"),limites[::,4:6],fmt="%s")

    def desenharCaixas(self,limites):
        im = self.imRes.matriz.copy()
        im = im.astype('uint8')
        label = self.ui.janela_img_res


        novaIm = np.array(np.zeros([im.shape[0],im.shape[1],3]),dtype='uint8')
        for i in range(3):
            novaIm[::,::,i] = im.copy()
        qim = QImage(novaIm, novaIm.shape[1], novaIm.shape[0], novaIm.strides[0], QImage.Format_RGB888)

        #Criar objeto de pintura
        painter = QPainter()
        #Criar objeto para escolha de cores e intensidade
        pen = QPen()
        pen.setColor(QColor(255,0,0))
        #Criando o pixmap a partir da imagem
        pixmap = QPixmap.fromImage(qim)
        #Escalando a imagem até a dimensão do label mantendo proporções
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio)

        #Verificando dimensões do pixmap após o scaling
        col = pixmap.width()
        lin = pixmap.height()
        #Definindo fator proporção entre o tamanho da imagem e o tamanho do pixmap
        fatorCol = col/im.shape[1]
        fatorLin = lin/im.shape[0]

        #Inicializando pintura utilizando pixmap como canva
        painter.begin(pixmap)
        #Selecionando padrão de cores e intensidade
        painter.setPen(pen)
        #Para cada região, gerar um ponto nas posições de centros de massa e uma linha nas orientação da região
        for i in range(limites.shape[0]):
            #Parâmetros para criação de retângulos
            x = limites[i,2]
            y = limites[i,0]
            height = limites[i,1] - limites[i,0]
            width = limites[i,3] - limites[i,2]
            #Desenhando retângulos
            rect = QRectF(fatorLin*x,fatorCol*y,fatorLin*width,fatorCol*height)
            painter.drawRect(rect)

        painter.end()

        #Setar o pixmap pintado como imagem mostrada no label da imagem resultante
        label.setPixmap(pixmap)

#############################################################################################################
####################                                                                     ####################
####################                           FUNÇÕES DE MENU                           ####################
####################                                                                     ####################
#############################################################################################################

    def abrirImagem1(self):
        #Tenta abrir uma imagem
        filename = QFileDialog.getOpenFileName(self)
        #Caso a abertura do arquivo seja cancelada, não executar restante
        if filename[0] == '':
            return

        #abre e guarda a imagem selecionada em self.im1
        aux = cv2.imread(filename[0])
        if aux is None:
            QMessageBox.warning(self, "Erro", "Caminho até imagem inválido.")
            return
        self.im1.matriz = cv2.cvtColor(aux,cv2.COLOR_RGB2BGR)
        self.im1.matriz = self.im1.matriz.astype('int')
        self.im1.tipo = 1
        self.im1.bordasAbs = None
        self.im1.bordasSentido = None
        self.im1.histograma = None
        self.im1.mapaRegioes = None
        self.im1.numRegioes = 0
        self.im1.regioes = None
        #atualiza o lavel da imagem 2
        self.atualizarImagem('im1')

        self.desativarSlider()

    def abrirImagem2(self):
        #Tenta abrir uma imagem
        filename = QFileDialog.getOpenFileName(self)
        #Caso a abertura do arquivo seja cancelada, não executar restante
        if filename[0] == '':
            return

        #abre e guarda a imagem selecionada em self.im2
        aux = cv2.imread(filename[0])
        if aux is None:
            QMessageBox.warning(self, "Erro", "Caminho até imagem inválido.")
            return
        self.im2.matriz = cv2.cvtColor(aux,cv2.COLOR_RGB2BGR)
        self.im2.matriz = self.im2.matriz.astype('int')
        self.im2.tipo = 1
        self.im2.bordasAbs = None
        self.im2.bordasSentido = None
        self.im2.histograma = None
        self.im2.mapaRegioes = None
        self.im2.numRegioes = 0
        self.im2.regioes = None
        #atualiza o label da imagem 2
        self.atualizarImagem('im2')

    def salvarImagem(self):
        #Caso o conteúdo da imagem resultante seja vazio, não executar restante
        if self.imRes.matriz is None:
            return

        filename = QFileDialog.getSaveFileName(self,"Salvar arquivo",".png","Imagens (*.png *.xpm *.jpg)")
        #Caso salvamento do arquivo seja cancelado, não executar restante
        if filename[0] == '':
            return

        self.imRes.matriz = self.imRes.matriz.astype('uint8')

        #Se imagem for RGB
        if self.imRes.tipo == 1:
            cv2.imwrite(filename[0],cv2.cvtColor(self.imRes.matriz,cv2.COLOR_RGB2BGR))
        #Se imagem for Mono
        elif self.imRes.tipo == 2 or self.imRes.tipo == 3:
            aux = np.zeros(self.imRes.matriz.shape + tuple([3]))
            for i in range(3):
                aux[:,:,i] = self.imRes.matriz
            cv2.imwrite(filename[0],aux)

    def apagaAtributosImRes(self,manter):
        if manter == 'nada':
            self.imRes.bordasAbs = None
            self.imRes.bordasSentido = None
            self.imRes.histograma = None
            self.imRes.mapaRegioes = None
            self.imRes.numRegioes = 0
            self.imRes.regioes = None
        elif manter == 'bordas':
            self.imRes.histograma = None
            self.imRes.mapaRegioes = None
            self.imRes.numRegioes = 0
            self.imRes.regioes = None
        elif manter == 'histograma':
            self.imRes.bordasAbs = None
            self.imRes.bordasSentido = None
            self.imRes.mapaRegioes = None
            self.imRes.numRegioes = 0
            self.imRes.regioes = None
        elif manter == 'segmentacao':
            self.imRes.bordasAbs = None
            self.imRes.bordasSentido = None
            self.imRes.histograma = None
            self.imRes.regioes = None
        elif manter == 'extracao':
            self.imRes.bordasAbs = None
            self.imRes.bordasSentido = None
            self.imRes.histograma = None

    def usarResultado(self):
        #Se contepudo da imagem resultante for vazio
        if self.imRes.matriz is None:
            QMessageBox.warning(self, "Erro", "Não existe imagem resultado.")
        else:
            #Copia imagem resultante para imagem 1
            self.im1.matriz = self.imRes.matriz.copy()
            self.im1.tipo = self.imRes.tipo
            if self.imRes.bordasAbs is not None:
                self.im1.bordasAbs = self.imRes.bordasAbs.copy()
            else:
                self.im1.bordasAbs = None

            if self.imRes.bordasSentido is not None:
                self.im1.bordasSentido = self.imRes.bordasSentido.copy()
            else:
                self.im1.bordasSentido = None

            if self.imRes.histograma is not None:
                self.im1.histograma = self.imRes.histograma.copy()
            else:
                self.im1.histograma = None

            if self.imRes.mapaRegioes is not None:
                self.im1.mapaRegioes = self.imRes.mapaRegioes.copy()
            else:
                self.im1.mapaRegioes = None

            self.im1.numRegioes = self.imRes.numRegioes

            if self.imRes.regioes is not None:
                self.im1.regioes = copy.deepcopy(self.imRes.regioes)

            #Atualiza labal da imagem 1
            self.atualizarImagem('im1')

            self.desativarSlider()

    def moverParaAuxiliar(self):
        #Se contepudo da imagem resultante for vazio
        if self.imRes.matriz is None:
            QMessageBox.warning(self, "Erro", "Não existe imagem resultado.")
        else:
            #Copia imagem resultante para imagem aux
            self.imAux.matriz = self.imRes.matriz.copy()
            self.imAux.tipo = self.imRes.tipo
            if self.imRes.bordasAbs is not None:
                self.imAux.bordasAbs = self.imRes.bordasAbs.copy()
            else:
                self.imAux.bordasAbs = None

            if self.imRes.bordasSentido is not None:
                self.imAux.bordasSentido = self.imRes.bordasSentido.copy()
            else:
                self.imAux.bordasSentido = None

            if self.imRes.histograma is not None:
                self.imAux.histograma = self.imRes.histograma.copy()
            else:
                self.imAux.histograma = None

            if self.imRes.mapaRegioes is not None:
                self.imAux.mapaRegioes = self.imRes.mapaRegioes.copy()
            else:
                self.imAux.mapaRegioes = None

            self.imAux.numRegioes = self.imRes.numRegioes

            if self.imRes.regioes is not None:
                self.imAux.regioes = copy.deepcopy(self.imRes.regioes)

            #Atualiza labal da imagem aux
            self.atualizarImagem('imAux')

    def atualizarImagem(self, tipo):
        #tipo 1: pega informações da imagem 1
        if tipo == 'im1':
            im = self.im1.matriz.copy()
            tip = self.im1.tipo
            label = self.ui.janela_img1
        #tipo res: pega informações da imagem resultante
        elif tipo == 'imRes':
            im = self.imRes.matriz.copy()
            tip = self.imRes.tipo
            label = self.ui.janela_img_res
        #tipo 2: pega informações da imagem 2
        elif tipo == 'im2':
            im = self.im2.matriz.copy()
            tip = self.im2.tipo
            label = self.ui.janela_img2
        #tipo aux: pega informações da imagem auxiliar
        else:
            im = self.imAux.matriz.copy()
            tip = self.imAux.tipo
            label = self.ui.janela_img_aux

        im = im.astype('uint8')
        #Caso o sistema de cor utilizado seja RGB
        if tip == 1:
            #Cria uma QImage com os canais RGB
            qim = QImage(im, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
            #qim = QImage(cv2.cvtColor(im,cv2.COLOR_RGB2BGR), im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
        #Caso o sistema de cor seja não seja RGB
        elif tip == 2 or tip == 3:
            novaIm = np.array(np.zeros([im.shape[0],im.shape[1],3]),dtype='uint8')
            for i in range(3):
                novaIm[::,::,i] = im.copy()
            qim = QImage(novaIm, novaIm.shape[1], novaIm.shape[0], novaIm.strides[0], QImage.Format_RGB888)
            #qim = QImage(im, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
            #qim.setColorTable([qRgb(i,i,i) for i in range(256)])

        #Cria um Pixmap a partir do QImage
        pixmap = QPixmap.fromImage(qim)

        #Redimensiona o pixmap pra o tamanho do label
        #pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio)

        #Atualiza o label
        label.setPixmap(pixmap)

    def ativaSlider(self,funcao,nome,maximo,valorinicial,minimo,passo):

        #Muda qual função vai utilizar o slider
        self.sliderflag = funcao
        #Muda as propriedades do slider
        self.ui.Slider.setMinimum(minimo)
        self.ui.Slider.setMaximum(maximo)
        self.ui.Slider.setSingleStep(passo)
        self.ui.Slider.setValue(valorinicial)
        self.ui.Slider.setSliderPosition(valorinicial)
        #Muda os Labels do slider
        self.ui.label_Slider.setText(nome)
        self.ui.label_Slider_max.setText(str(maximo))
        self.ui.label_Slider_valor.setText(str(valorinicial))
        self.ui.label_Slider_min.setText(str(minimo))
        self.ui.Slider.setEnabled(True)

    def desativarSlider(self):

        #Avisa que nenhuma função está usando slider
        self.sliderflag = 0
        #Desativa slider e muda os labels
        self.ui.Slider.setEnabled(False)
        self.ui.label_Slider.setText('Slider')
        self.ui.label_Slider_max.setText('max')
        self.ui.label_Slider_valor.setText('val')
        self.ui.label_Slider_min.setText('min')

    def atualizarSlider(self):
        #Atualiza valor que está sendo mostrado com base na mudança do slide
        self.ui.label_Slider_valor.setText(str(self.ui.Slider.value()))

        #slidflag == 1 usa função de converter para binário
        if self.sliderflag == 1:
            self.converterBinMenu()

    def converterParaCinzaMenu(self):
        #Se não houver imagem carregada, tenta abrir uma
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Se o conteúdo da imagem 1 for vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Copia imagem 1 para imagem resultante
        self.imRes.matriz = self.im1.matriz.copy()
        self.imRes.tipo = self.im1.tipo

        #Se for RGB
        if self.imRes.tipo == 1:
            self.imRes.matriz, self.imRes.tipo = self.converterParaCinza(self.imRes.matriz,self.imRes.tipo)

        #Converte valores para inteiros
        self.imRes.matriz = self.imRes.matriz.astype('int')
        #Muda label da imagem resultante
        self.atualizarImagem('imRes')

        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def converterBinMenu(self):
        #funcionamento da conversão
        if self.sliderflag == 1:

            value = self.ui.Slider.value()
            self.imRes.matriz = self.im1.matriz.copy()
            #self.imRes.tipo = self.im1_tipo #não precisa porque já é Mono

            self.imRes.matriz, self.imRes.tipo = self.converterBin(value,self.imRes.matriz,self.imRes.tipo)
            #self.imRes.matriz[self.imRes.matriz > value] = 255
            #self.imRes.matriz[self.imRes.matriz <= value] = 0

            self.imRes.matriz = self.imRes.matriz.astype('int')
            self.atualizarImagem('imRes')

            #Apagar atributos que não fazem parte do bloco
            self.apagaAtributosImRes('nada')

        #Preparando uso da função
        else:
            #Caso não haja imagem carregada
            if self.im1.matriz is None:
                self.abrirImagem1()
            #Caso  o conteúdo da imagem seja vazio, não executar o restante
            if self.im1.matriz is None:
                return

            #Caso a imagem esteja em RGB
            if self.im1.tipo == 1:
                self.converterParaCinzaMenu()
                self.usarResultado()

            #Ativa Slider
            self.ativaSlider(1,'Limiar',255,127,0,1)
            self.atualizarSlider()

    def somarMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Escolher tipo de soma
        minhasOpcoes = ["Escalar","Imagem 2"]
        opcao, flag = QInputDialog.getItem(self,"Soma","Somar com:",minhasOpcoes)
        #Caso cancele, parar código
        if not(flag):
            return

        #Se soma escolhida foi escalar
        if opcao == minhasOpcoes[0]:
            escalar, flag = QInputDialog.getDouble(self,"Soma Com Escalar","Somar com:")
            #Caso cancele, parar código
            if not(flag):
                return

            img = self.somarEscalar(self.im1.matriz.copy(),escalar)
            tipo = self.im1.tipo

        #Se soma escolhida foi com imagem 2
        else:
            #Caso não haja imagem 2 carregada
            if self.im2.matriz is None:
                self.abrirImagem2()
            #Caso  o conteúdo da imagem2 seja vazio, não executar o restante
            if self.im2.matriz is None:
                return

            #Testa se imagem 2 tem mesma dimensão que imagem 1
            if self.im1.matriz.shape != self.im2.matriz.shape:
                QMessageBox.warning(self, "Erro", "Imagens com dimensões diferentes!")
                return

            img = self.somarImagens(self.im1.matriz.copy(),self.im2.matriz.copy())
            tipo = self.im1.tipo

        #Pergunta se é necessário escalonar
        pergunta = MyQuestion("Pergunta:","Escalonar imagem?")
        if pergunta.exec_():
            img = self.escalonarImagem(img)
        #Truncamento dos valores
        else:
            img = self.truncarImagem(img)

        self.imRes.matriz = img
        self.imRes.tipo = tipo
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def subtrairMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Escolher tipo de subtração
        minhasOpcoes = ["Escalar","Imagem 2"]
        opcao, flag = QInputDialog.getItem(self,"Subtração","Subtrair com:",minhasOpcoes)
        #Caso cancele, parar código
        if not(flag):
            return

        #Se subtração escolhida foi escalar
        if opcao == minhasOpcoes[0]:
            escalar, flag = QInputDialog.getDouble(self,"Subtração Com Escalar","Subtrair com:")
            #Caso cancele, parar código
            if not(flag):
                return

            img = self.subtrairEscalar(self.im1.matriz.copy(),escalar)
            tipo = self.im1.tipo

        #Se subtração escolhida foi com imagem 2
        else:
            #Caso não haja imagem 2 carregada
            if self.im2.matriz is None:
                self.abrirImagem2()
            #Caso  o conteúdo da imagem2 seja vazio, não executar o restante
            if self.im2.matriz is None:
                return

            #Testa se imagem 2 tem mesma dimensão que imagem 1
            if self.im1.matriz.shape != self.im2.matriz.shape:
                QMessageBox.warning(self, "Erro", "Imagens com dimensões diferentes!")
                return

            img = self.subtrairImagens(self.im1.matriz.copy(),self.im2.matriz.copy())
            tipo = self.im1.tipo

        #Pergunta se é necessário escalonar
        pergunta = MyQuestion("Pergunta:","Escalonar imagem?")
        if pergunta.exec_():
            img = self.escalonarImagem(img)
        #Truncamento dos valores
        else:
            img = self.truncarImagem(img)

        self.imRes.matriz = img
        self.imRes.tipo = tipo
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def multiplicarMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        minhasOpcoes = ["Escalar","Imagem 2"]
        escalar, flag = QInputDialog.getDouble(self,"Multiplicação","Multiplicar com:")
        if not(flag):
            return

        img = self.multiplicarEscalar(self.im1.matriz.copy(),escalar)
        tipo = self.im1.tipo

        #Pergunta se é necessário escalonar
        pergunta = MyQuestion("Pergunta:","Escalonar imagem?")
        if pergunta.exec_():
            img = self.escalonarImagem(img)
        #Truncamento dos valores
        else:
            img = self.truncarImagem(img)

        self.imRes.matriz = img
        self.imRes.tipo = tipo
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def dividirMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        minhasOpcoes = ["Escalar","Imagem 2"]
        escalar, flag = QInputDialog.getDouble(self,"Divisão","Dividir com:")
        if not(flag):
            return

        img = self.dividirEscalar(self.im1.matriz.copy(),escalar)
        tipo = self.im1.tipo

        #Pergunta se é necessário escalonar
        pergunta = MyQuestion("Pergunta:","Escalonar imagem?")
        if pergunta.exec_():
            img = self.escalonarImagem(img)
        #Truncamento dos valores
        else:
            img = self.truncarImagem(img)

        self.imRes.matriz = img
        self.imRes.tipo = tipo
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def fazerNotMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso imagem 1 não seja binária
        if self.im1.tipo != 3:
            #Converter para Mono
            self.im1.matriz, self.im1.tipo = self.converterParaCinza(self.im1.matriz, self.im1.tipo)
            #Transformar em binário
            self.im1.matriz, self.im1.tipo = self.converterBin(127,self.im1.matriz,self.im1.tipo)
            self.atualizarImagem('im1')

        #Operação de NOT
        self.imRes.matriz = self.fazerNot(self.im1.matriz.copy())
        self.imRes.tipo = self.im1.tipo

        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def fazerAndMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Caso não haja imagem 2 carregada
        if self.im2.matriz is None:
            self.abrirImagem2()
        #Caso  o conteúdo da imagem2 seja vazio, não executar o restante
        if self.im2.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()
        #Testa se imagem 2 tem mesma dimensão que imagem 1
        if self.im1.matriz.shape[0:2] != self.im2.matriz.shape[0:2]:
            QMessageBox.warning(self, "Erro", "Imagens com dimensões diferentes!")
            return

        #Caso imagem 1 ou imagem 2 não sejam binárias
        if self.im1.tipo != 3 or self.im2.tipo != 3:
            #Converter para Mono
            self.im1.matriz, self.im1.tipo = self.converterParaCinza(self.im1.matriz, self.im1.tipo)
            self.im2.matriz, self.im2.tipo = self.converterParaCinza(self.im2.matriz, self.im2.tipo)
            #Transformar em binário
            self.im1.matriz, self.im1.tipo = self.converterBin(127,self.im1.matriz,self.im1.tipo)
            self.im2.matriz, self.im2.tipo = self.converterBin(127,self.im2.matriz,self.im2.tipo)
            self.atualizarImagem('im1')
            self.atualizarImagem('im2')

        #Operação de AND
        self.imRes.matriz = self.fazerAnd(self.im1.matriz,self.im2.matriz)
        self.imRes.tipo = self.im1.tipo
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def fazerOrMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Caso não haja imagem 2 carregada
        if self.im2.matriz is None:
            self.abrirImagem2()
        #Caso  o conteúdo da imagem2 seja vazio, não executar o restante
        if self.im2.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()
        #Testa se imagem 2 tem mesma dimensão que imagem 1
        if self.im1.matriz.shape[0:2] != self.im2.matriz.shape[0:2]:
            QMessageBox.warning(self, "Erro", "Imagens com dimensões diferentes!")
            return

        #Caso imagem 1 ou imagem 2 não sejam binárias
        if self.im1.tipo != 3 or self.im2.tipo != 3:
            #Converter para Mono
            self.im1.matriz, self.im1.tipo = self.converterParaCinza(self.im1.matriz, self.im1.tipo)
            self.im2.matriz, self.im2.tipo = self.converterParaCinza(self.im2.matriz, self.im2.tipo)
            #Transformar em binário
            self.im1.matriz, self.im1.tipo = self.converterBin(127,self.im1.matriz,self.im1.tipo)
            self.im2.matriz, self.im2.tipo = self.converterBin(127,self.im2.matriz,self.im2.tipo)
            self.atualizarImagem('im1')
            self.atualizarImagem('im2')

        #Operação de OR
        self.imRes.matriz = self.fazerOr(self.im1.matriz,self.im2.matriz)
        self.imRes.tipo = self.im1.tipo
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def fazerXorMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Caso não haja imagem 2 carregada
        if self.im2.matriz is None:
            self.abrirImagem2()
        #Caso  o conteúdo da imagem2 seja vazio, não executar o restante
        if self.im2.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()
        #Testa se imagem 2 tem mesma dimensão que imagem 1
        if self.im1.matriz.shape[0:2] != self.im2.matriz.shape[0:2]:
            QMessageBox.warning(self, "Erro", "Imagens com dimensões diferentes!")
            return

        #Caso imagem 1 ou imagem 2 não sejam binárias
        if self.im1.tipo != 3 or self.im2.tipo != 3:
            #Converter para Mono
            self.im1.matriz, self.im1.tipo = self.converterParaCinza(self.im1.matriz, self.im1.tipo)
            self.im2.matriz, self.im2.tipo = self.converterParaCinza(self.im2.matriz, self.im2.tipo)
            #Transformar em binário
            self.im1.matriz, self.im1.tipo = self.converterBin(127,self.im1.matriz,self.im1.tipo)
            self.im2.matriz, self.im2.tipo = self.converterBin(127,self.im2.matriz,self.im2.tipo)
            self.atualizarImagem('im1')
            self.atualizarImagem('im2')

        #Operação de XOR
        self.imRes.matriz = self.fazerXor(self.im1.matriz.copy(),self.im2.matriz.copy())
        self.imRes.tipo = self.im1.tipo
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def transladarMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Chamo a caixa de diálogo
        translacao = MinhaTranslacao()
        if translacao.exec_():
            #Recebe valores da caixa de diálogo
            dx, dy = translacao.getValores()
            #implementa translação
            self.imRes.matriz = self.transladar(self.im1.matriz.copy(),self.im1.tipo,dx,dy)
            self.imRes.tipo = self.im1.tipo

            #Atualiza imagem resultante
            self.atualizarImagem("imRes")
            #Apagar atributos que não fazem parte do bloco
            self.apagaAtributosImRes('nada')
        #Parar o método se caixa de diálogo não for confirmada
        else:
            return

    def rotacionarMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Verifica dimensões da matriz
        linhas = self.im1.matriz.shape[0]
        colunas = self.im1.matriz.shape[1]

        #Chamo a caixa de diálogo
        rotacao = MinhaRotacao(linhas, colunas)
        if rotacao.exec_():
            #Recebe valores da caixa de diálogo
            angulo, coordX, coordY = rotacao.getValores()
            #implementa rotação
            self.imRes.matriz = self.rotacionar(self.im1.matriz.copy(),self.im1.tipo,angulo,coordX,coordY)
            self.imRes.tipo = self.im1.tipo

            #Atualiza imagem resultante
            self.atualizarImagem("imRes")
            #Apagar atributos que não fazem parte do bloco
            self.apagaAtributosImRes('nada')
        #Parar o método se caixa de diálogo não for confirmada
        else:
            return

    def escalonarMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Verifica dimensões da matriz
        linhas = self.im1.matriz.shape[0]
        colunas = self.im1.matriz.shape[1]

        #Chamo a caixa de diálogo
        escalonamento = MeuEscalonamento(linhas, colunas)
        if escalonamento.exec_():
            #Recebe valores da caixa de diálogo
            escalarX, escalarY, coordX, coordY = escalonamento.getValores()
            #implementa escalonamento
            self.imRes.matriz = self.escalonar(self.im1.matriz.copy(),self.im1.tipo,escalarX,escalarY,coordX,coordY)
            self.imRes.tipo = self.im1.tipo

            #Atualiza imagem resultante
            self.atualizarImagem("imRes")
            #Apagar atributos que não fazem parte do bloco
            self.apagaAtributosImRes('nada')

        #Parar o método se caixa de diálogo não for confirmada
        else:
            return

    def usarDerivativo1Menu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 2 or self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Acha intensidades e sentido das bordas (sentido segue orientação x vertical e y horizontal)
        self.imRes.bordasAbs, self.imRes.bordasSentido = self.detectarDerivativo(imagem,1)

        #Define Limiar
        limiar, flag = QInputDialog.getInt(self,"Derivativo 1","Limiar:")
        #Caso cancele, parar código
        if not(flag):
            return

        #Acha as bordas com base no limiar
        self.imRes.matriz, self.imRes.tipo = self.converterBin(limiar,self.imRes.bordasAbs.copy(),2)
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('bordas')

    def usarDerivativo2Menu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 2 or self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Acha intensidades e sentido das bordas (sentido segue orientação x vertical e y horizontal)
        self.imRes.bordasAbs, self.imRes.bordasSentido = self.detectarDerivativo(imagem,2)

        #Define Limiar
        limiar, flag = QInputDialog.getInt(self,"Derivativo 2","Limiar:")
        #Caso cancele, parar código
        if not(flag):
            return

        #Acha as bordas com base no limiar
        self.imRes.matriz, self.imRes.tipo = self.converterBin(limiar,self.imRes.bordasAbs.copy(),2)
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('bordas')

    def usarSobelMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()


        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 2 or self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Acha intensidades e sentido das bordas
        self.imRes.bordasAbs, self.imRes.bordasSentido = self.detectarSobel(imagem)

        #Define Limiar
        limiar, flag = QInputDialog.getInt(self,"Sobel","Limiar:")
        #Caso cancele, parar código
        if not(flag):
            return

        #Acha as bordas com base no limiar
        self.imRes.matriz, self.imRes.tipo = self.converterBin(limiar,self.imRes.bordasAbs.copy(),2)
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('bordas')

    def usarKirschMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 2 or self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Acha intensidades e sentido das bordas
        self.imRes.bordasAbs, self.imRes.bordasSentido = self.detectarKirsch(imagem)

        #Define Limiar
        limiar, flag = QInputDialog.getInt(self,"Kirsch","Limiar:")
        #Caso cancele, parar código
        if not(flag):
            return

        #Acha as bordas com base no limiar
        self.imRes.matriz, self.imRes.tipo = self.converterBin(limiar,self.imRes.bordasAbs.copy(),2)
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('bordas')

    def gerarHistogramaMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 2 or self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Calcula o histograma da imagem 1
        self.im1.histograma = self.histograma(imagem)
        #Plota o histograma da imagem 1
        self.plotarHistograma(self.im1.histograma,"Imagem 1")

    def fazerAutoescalaMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Escalona todos os canais da imagem 1
        self.imRes.matriz = self.escalonarImagem(self.im1.matriz.copy())
        self.imRes.tipo = self.im1.tipo
        self.atualizarImagem("imRes")

        #Calcula o histograma da imagem resultante
        self.imRes.histograma = self.histograma(self.imRes.matriz.copy())
        #Plota o histograma da imagem resultante
        self.plotarHistograma(self.imRes.histograma.copy(),"Imagem Resultante")
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('histograma')

    def equalizarHistogramaMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 2 or self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Calcula o histograma da imagem 1
        self.im1.histograma = self.histograma(imagem)

        #Fazendo operação de equalização e mostrando na imagem resultante
        cdfAnterior, self.imRes.matriz = self.equalizarHistograma(self.im1.histograma.copy(),imagem)
        self.imRes.tipo = tipo
        self.atualizarImagem('imRes')

        #Calcula o histograma da imagem resultante
        self.imRes.histograma = self.histograma(self.imRes.matriz.copy())
        #Plota o histograma da imagem resultante
        #self.plotarHistograma(self.imRes.histograma,"Imagem Resultante")

        #Plotando o comparativo das funções cumulativas
        self.plotarCumulativa(cdfAnterior,self.imRes.histograma.copy())
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('histograma')

    def limiarizarGlobalMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 2 or self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Calcula o histograma da imagem 1
        self.im1.histograma = self.histograma(imagem)

        #Calculo do limiar global
        limiar = self.limiarizarIterativo(self.im1.histograma.copy(),imagem)

        #print(limiar)

        #Conversão para binário utilizando limiar global
        self.imRes.matriz, self.imRes.tipo = self.converterBin(limiar,imagem,tipo)
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def limiarizarOtsuMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 2 or self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Calcula o histograma da imagem 1
        self.im1.histograma = self.histograma(imagem)

        #Calculo do limiar de Otsu
        limiar = self.limiarizarOtsu(self.im1.histograma.copy(),imagem)

        #print(limiar)

        #Conversão para binário utilizando limiar de Otsu
        self.imRes.matriz, self.imRes.tipo = self.converterBin(limiar,imagem,tipo)
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def filtrarMediaMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Chamo a caixa de diálogo
        filtroMediaDialog = MeuFiltroMedia()
        if filtroMediaDialog.exec_():
            #Recebe valores da caixa de diálogo
            mLinhas, mColunas = filtroMediaDialog.getValores()

            #Uso do filtro de média
            self.imRes.matriz = self.filtrarMedia(self.im1.matriz.copy(),self.im1.tipo,mLinhas,mColunas)
            self.imRes.tipo = self.im1.tipo

            #Atualiza imagem resultante
            self.atualizarImagem("imRes")
            #Apagar atributos que não fazem parte do bloco
            self.apagaAtributosImRes('nada')
        #Parar o método se caixa de diálogo não for confirmada
        else:
            return

    def filtrarMedianaMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Uso do filtro de mediana
        self.imRes.matriz = self.filtrarMediana(self.im1.matriz.copy(),self.im1.tipo)
        self.imRes.tipo = self.im1.tipo

        #Atualiza imagem resultante
        self.atualizarImagem("imRes")
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def filtrarPassaAltaMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Uso do filtro passa altas
        self.imRes.matriz = self.filtrarPassaAlta(self.im1.matriz.copy(),self.im1.tipo)
        self.imRes.tipo = self.im1.tipo

        #Atualiza imagem resultante
        self.atualizarImagem("imRes")
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def filtrarLaplacianoMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Uso do filtro com máscara Laplaciano
        self.imRes.matriz = self.filtrarLaplaciano(self.im1.matriz.copy(),self.im1.tipo)
        self.imRes.tipo = self.im1.tipo

        #Atualiza imagem resultante
        self.atualizarImagem("imRes")
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def dilatarMorfoMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
            imagem, tipo = self.converterBin(126,imagem,tipo);
        elif self.im1.tipo == 2:
            imagem, tipo = self.converterBin(126,self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Pergunta se as operações devem ser feitas em pixels pretos ou brancos
        pergunta = MyQuestion("Pergunta:","Fazer operações em pixels pretos?")
        if pergunta.exec_():
            cor = 1
        else:
            cor = 0

        #Perguntar dimensão do elemento estruturante
        escalar, flag = QInputDialog.getInt(self,"Dilatação","Digite a dimensão do elemento estruturante:")
        if not(flag):
            return
        matriz = np.zeros([escalar,escalar])

        #Código só aceita dimensão ímpar
        if not escalar%2:
            QMessageBox.warning(self, "Atenção", "Colocar uma dimensão ímpar")
            return

        #Pergunta se é necessário escalonar
        pergunta2 = MyQuestion("Pergunta:","Deseja setar todos os pixels?")
        #Caso não deseje setar todos os pixels
        if not pergunta2.exec_():
            #Chamo a caixa de diálogo
            elementoEstruturanteMenu = MeuElementoEstruturante(escalar)
            #Caso seja cancelado, parar código
            if not elementoEstruturanteMenu.exec_():
                return
            #Armazenando a matriz contendo o elemento estruturante
            matriz = elementoEstruturanteMenu.getMatriz()
        #Caso deseje setar todos os pixels
        else:
            matriz[::,::] = 1

        #Executando operação de dilatação
        self.imRes.matriz = self.fazerDilatacao(imagem,matriz,cor)
        self.imRes.tipo =  3

        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def erodirMorfoMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
            imagem, tipo = self.converterBin(126,imagem,tipo);
        elif self.im1.tipo == 2:
            imagem, tipo = self.converterBin(126,self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Pergunta se as operações devem ser feitas em pixels pretos ou brancos
        pergunta = MyQuestion("Pergunta:","Fazer operações em pixels pretos?")
        if pergunta.exec_():
            cor = 1
        else:
            cor = 0

        #Perguntar dimensão do elemento estruturante
        escalar, flag = QInputDialog.getInt(self,"Erosão","Digite a dimensão do elemento estruturante:")
        if not(flag):
            return
        matriz = np.zeros([escalar,escalar])

        #Código só aceita dimensão ímpar
        if not escalar%2:
            QMessageBox.warning(self, "Atenção", "Colocar uma dimensão ímpar")
            return

        #Pergunta se é necessário escalonar
        pergunta2 = MyQuestion("Pergunta:","Deseja setar todos os pixels?")
        #Caso não deseje setar todos os pixels
        if not pergunta2.exec_():
            #Chamo a caixa de diálogo
            elementoEstruturanteMenu = MeuElementoEstruturante(escalar)
            #Caso seja cancelado, parar código
            if not elementoEstruturanteMenu.exec_():
                return
            #Armazenando a matriz contendo o elemento estruturante
            matriz = elementoEstruturanteMenu.getMatriz()
        #Caso deseje setar todos os pixels
        else:
            matriz[::,::] = 1

        #Executando operação de erosão
        self.imRes.matriz = self.fazerErosao(imagem,matriz,cor)
        self.imRes.tipo =  3

        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def abrirMorfoMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
            imagem, tipo = self.converterBin(126,imagem,tipo);
        elif self.im1.tipo == 2:
            imagem, tipo = self.converterBin(126,self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Pergunta se as operações devem ser feitas em pixels pretos ou brancos
        pergunta = MyQuestion("Pergunta:","Fazer operações em pixels pretos?")
        if pergunta.exec_():
            cor = 1
        else:
            cor = 0

        #Perguntar dimensão do elemento estruturante
        escalar, flag = QInputDialog.getInt(self,"Abertura","Digite a dimensão do elemento estruturante:")
        if not(flag):
            return
        matriz = np.zeros([escalar,escalar])

        #Código só aceita dimensão ímpar
        if not escalar%2:
            QMessageBox.warning(self, "Atenção", "Colocar uma dimensão ímpar")
            return

        #Pergunta se é necessário escalonar
        pergunta2 = MyQuestion("Pergunta:","Deseja setar todos os pixels?")
        #Caso não deseje setar todos os pixels
        if not pergunta2.exec_():
            #Chamo a caixa de diálogo
            elementoEstruturanteMenu = MeuElementoEstruturante(escalar)
            #Caso seja cancelado, parar código
            if not elementoEstruturanteMenu.exec_():
                return
            #Armazenando a matriz contendo o elemento estruturante
            matriz = elementoEstruturanteMenu.getMatriz()
        #Caso deseje setar todos os pixels
        else:
            matriz[::,::] = 1

        #Executando operação de abertura
        self.imRes.matriz = self.fazerAbertura(imagem,matriz,cor)
        self.imRes.tipo =  3

        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def fecharMorfoMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
            imagem, tipo = self.converterBin(126,imagem,tipo);
        elif self.im1.tipo == 2:
            imagem, tipo = self.converterBin(126,self.im1.matriz.copy(), self.im1.tipo)
        elif self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Pergunta se as operações devem ser feitas em pixels pretos ou brancos
        pergunta = MyQuestion("Pergunta:","Fazer operações em pixels pretos?")
        if pergunta.exec_():
            cor = 1
        else:
            cor = 0

        #Perguntar dimensão do elemento estruturante
        escalar, flag = QInputDialog.getInt(self,"Fechamento","Digite a dimensão do elemento estruturante:")
        if not(flag):
            return
        matriz = np.zeros([escalar,escalar])

        #Código só aceita dimensão ímpar
        if not escalar%2:
            QMessageBox.warning(self, "Atenção", "Colocar uma dimensão ímpar")
            return

        #Pergunta se é necessário escalonar
        pergunta2 = MyQuestion("Pergunta:","Deseja setar todos os pixels?")
        #Caso não deseje setar todos os pixels
        if not pergunta2.exec_():
            #Chamo a caixa de diálogo
            elementoEstruturanteMenu = MeuElementoEstruturante(escalar)
            #Caso seja cancelado, parar código
            if not elementoEstruturanteMenu.exec_():
                return
            #Armazenando a matriz contendo o elemento estruturante
            matriz = elementoEstruturanteMenu.getMatriz()
        #Caso deseje setar todos os pixels
        else:
            matriz[::,::] = 1

        #Executando operação de fechamento
        self.imRes.matriz = self.fazerFechamento(imagem,matriz,cor)
        self.imRes.tipo =  3

        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('nada')

    def segmentacaoPropostaMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
            imagem, tipo = self.converterBin(self.limiarizarOtsu(self.histograma(imagem),imagem),imagem,tipo)
            #self.im1.matriz, self.im1.tipo = imagem.copy(), tipo
            #self.atualizarImagem('im1')
        elif self.im1.tipo == 2:
            imagem, tipo = self.converterBin(self.limiarizarOtsu(self.histograma(self.im1.matriz.copy()),self.im1.matriz.copy()),self.im1.matriz.copy(), self.im1.tipo)
            #self.im1.matriz, self.im1.tipo = imagem.copy(), tipo
            #self.atualizarImagem('im1')
        elif self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        flag = 0;
        #Pergunta se as operações devem ser feitas em pixels pretos ou brancos
        pergunta = MyQuestion("Pergunta:","Fazer sementação para pixels escuros?")
        if pergunta.exec_():
            flag = 1
        #Caso seja pedido para segmentar pixels pretos
        if flag == 1:
            imagem = self.fazerNot(imagem)

        #Executar algoritmo de segmentação excluindo regiões com 0,1% do tamanho da imagem
        self.imRes.mapaRegioes,self.imRes.numRegioes = self.segmentacaoProposta(imagem,0.001)

        #Escalonando mapa de regiões para ser visível na matriz da imagem
        self.imRes.matriz = self.escalonarImagem(self.imRes.mapaRegioes)
        self.imRes.tipo = 2

        #print(self.imRes.mapaRegioes)
        #print("Regiões:",self.imRes.numRegioes)
        self.atualizarImagem('imRes')
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('segmentacao')

    def extrairCaracteristicasMenu(self):
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
            imagem, tipo = self.converterBin(self.limiarizarOtsu(self.histograma(imagem),imagem),imagem,tipo)
            #self.im1.matriz, self.im1.tipo = imagem.copy(), tipo
            #self.atualizarImagem('im1')
        elif self.im1.tipo == 2:
            imagem, tipo = self.converterBin(self.limiarizarOtsu(self.histograma(self.im1.matriz.copy()),self.im1.matriz.copy()),self.im1.matriz.copy(), self.im1.tipo)
            #self.im1.matriz, self.im1.tipo = imagem.copy(), tipo
            #self.atualizarImagem('im1')
        elif self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        #Caso imagem ainda não esteja segmentada
        if self.im1.numRegioes == 0:
            flag = 0;
            #Pergunta se as operações devem ser feitas em pixels pretos ou brancos
            pergunta = MyQuestion("Pergunta:","Fazer extração avaliando os pixels escuros?")
            if pergunta.exec_():
                flag = 1
            #Caso seja pedido para segmentar pixels pretos
            if flag == 1:
                imagem = self.fazerNot(imagem)
            #Executar algoritmo de segmentação excluindo regiões com 0,1% do tamanho da imagem
            self.imRes.mapaRegioes,self.imRes.numRegioes = self.segmentacaoProposta(imagem,0.001)

            if flag == 1:
                imagem = self.fazerNot(imagem)

        #Gerando matriz resultante igual a matriz da imagem 1
        self.imRes.matriz = self.im1.matriz.copy()
        self.imRes.tipo = self.im1.tipo

        #Criando uma lista de regiões
        self.imRes.regioes = []
        for i in range(self.imRes.numRegioes):
            mapa = self.imRes.mapaRegioes.copy()
            mapa[mapa != i+1] = 0
            mapa[mapa == i+1] = 1
            self.imRes.regioes.append(Regiao(i+1,mapa))

        #Executando função de extração de característica
        for i in range(self.imRes.numRegioes):
            self.extrairCaracteristicas(self.imRes.regioes[i])
            #print("Regiao",self.imRes.regioes[i].label)
            #print(self.imRes.regioes[i].mapa)
            #print("Area",self.imRes.regioes[i].area)
            #print("Centro de massa",self.imRes.regioes[i].centroDeMassa)
            #print("Orientacao",self.imRes.regioes[i].orientacao,"graus")
            #print("Comprimento",self.imRes.regioes[i].comprimento)
            #print("Largura",self.imRes.regioes[i].largura)
            #print("Extremos",self.imRes.regioes[i].extremos)
            #print("")

        #Abrir janela com informações de todas as regiões
        self.mostrarCaracteristicas(self.imRes.regioes)

        #Desenha orientações e centros de massa na imagem resultante
        self.desenharCaracteristicas(self.imRes.regioes)
        #Apagar atributos que não fazem parte do bloco
        self.apagaAtributosImRes('extracao')

    def desenharCaracteristicas(self,regioes):
        im = self.imRes.matriz.copy()
        im = im.astype('uint8')
        tip = self.imRes.tipo
        label = self.ui.janela_img_res

        #Caso o sistema de cor utilizado seja RGB
        if tip == 1:
            #Cria uma QImage com os canais RGB
            qim = QImage(im, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
            #qim = QImage(cv2.cvtColor(im,cv2.COLOR_RGB2BGR), im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
        #Caso o sistema de cor seja não seja RGB
        elif tip == 2 or tip == 3:
            novaIm = np.array(np.zeros([im.shape[0],im.shape[1],3]),dtype='uint8')
            for i in range(3):
                novaIm[::,::,i] = im.copy()
            qim = QImage(novaIm, novaIm.shape[1], novaIm.shape[0], novaIm.strides[0], QImage.Format_RGB888)

        #Criar objeto de pintura
        painter = QPainter()
        #Criar objeto para escolha de cores e intensidade
        pen = QPen()
        pen.setColor(QColor(255,0,0))
        #Criando o pixmap a partir da imagem
        pixmap = QPixmap.fromImage(qim)
        #Escalando a imagem até a dimensão do label mantendo proporções
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio)

        #Verificando dimensões do pixmap após o scaling
        col = pixmap.width()
        lin = pixmap.height()
        #Definindo fator proporção entre o tamanho da imagem e o tamanho do pixmap
        fatorCol = col/regioes[0].mapa.shape[1]
        fatorLin = lin/regioes[0].mapa.shape[0]

        #Inicializando pintura utilizando pixmap como canva
        painter.begin(pixmap)
        #Selecionando padrão de cores e intensidade
        painter.setPen(pen)
        #Para cada região, gerar um ponto nas posições de centros de massa e uma linha nas orientação da região
        for i in range(len(regioes)):
            #Definindo ponto e linha
            point = QPointF(fatorCol*regioes[i].centroDeMassa[0],fatorLin*regioes[i].centroDeMassa[1])
            line  = QLineF(fatorCol*regioes[i].extremos[1,0],fatorLin*regioes[i].extremos[0,0],fatorCol*regioes[i].extremos[1,1],fatorLin*regioes[i].extremos[0,1])
            #Desenhando ponto com 5x5 pixels
            pen.setWidth(5)
            painter.setPen(pen)
            painter.drawPoint(point)

            #Desenhando linha com 2 pixels
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(line)
        #Encerrar modo de pintura do pixmap
        painter.end()

        #Setar o pixmap pintado como imagem mostrada no label da imagem resultante
        label.setPixmap(pixmap)

    def mostrarCaracteristicas(self,regioes):
        self.janela = Ui_JanelaRegioes() #Cria uma janela para receber texto
        self.janela.caixaRegioes.setPlainText("")

        for i in range(len(regioes)):
            text = str(self.imRes.regioes[i].label)
            self.janela.caixaRegioes.appendPlainText("Região " + text)

            text = str(np.round(self.imRes.regioes[i].area,2))
            self.janela.caixaRegioes.appendPlainText("Área: " + text)

            text1 = str(np.round(self.imRes.regioes[i].centroDeMassa[0],2))
            text2 = str(np.round(self.imRes.regioes[i].centroDeMassa[1],2))
            self.janela.caixaRegioes.appendPlainText("Centro de Massa: [" + text1 + " , " + text2 + "]")

            text = str(np.round(self.imRes.regioes[i].orientacao,2))
            self.janela.caixaRegioes.appendPlainText("Orientação: " + text + " graus")

            text = str(np.round(self.imRes.regioes[i].comprimento,2))
            self.janela.caixaRegioes.appendPlainText("Comprimento: " + text)

            text = str(np.round(self.imRes.regioes[i].largura,2))
            self.janela.caixaRegioes.appendPlainText("Largura: " + text + "\n")

        self.janela.show()

    def transformadaDeHoughMenu(self):
        print("Transformada de Hough")

        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
            imagem, tipo = self.converterBin(self.limiarizarOtsu(self.histograma(imagem),imagem),imagem,tipo)
            #self.im1.matriz, self.im1.tipo = imagem.copy(), tipo
            #self.atualizarImagem('im1')
        elif self.im1.tipo == 2:
            imagem, tipo = self.converterBin(self.limiarizarOtsu(self.histograma(self.im1.matriz.copy()),self.im1.matriz.copy()),self.im1.matriz.copy(), self.im1.tipo)
            #self.im1.matriz, self.im1.tipo = imagem.copy(), tipo
            #self.atualizarImagem('im1')
        elif self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        matriz,angulo = self.transformadaDeHough(imagem)

        self.imRes.matriz = self.escalonarImagem(matriz.copy())
        #self.imRes.matriz = matriz.copy()
        self.imRes.tipo = 2

        self.atualizarImagem('imRes')

    def correcaoDeOrientacaoMenu(self):
        print("Correção de Orientação")
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
            imagem, tipo = self.converterBin(self.limiarizarOtsu(self.histograma(imagem),imagem),imagem,tipo)
            #self.im1.matriz, self.im1.tipo = imagem.copy(), tipo
            #self.atualizarImagem('im1')
        elif self.im1.tipo == 2:
            imagem, tipo = self.converterBin(self.limiarizarOtsu(self.histograma(self.im1.matriz.copy()),self.im1.matriz.copy()),self.im1.matriz.copy(), self.im1.tipo)
            #self.im1.matriz, self.im1.tipo = imagem.copy(), tipo
            #self.atualizarImagem('im1')
        elif self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo

        matriz,angulo = self.transformadaDeHough(imagem)
        self.imRes.matriz = self.novoRotacionar(self.im1.matriz.copy(),self.im1.tipo,-angulo)
        self.imRes.tipo = self.im1.tipo

        self.atualizarImagem('imRes')

    def ocrMenu(self):
        print("Execução do OCR")
        #Caso não haja imagem carregada
        if self.im1.matriz is None:
            self.abrirImagem1()
        #Caso  o conteúdo da imagem seja vazio, não executar o restante
        if self.im1.matriz is None:
            return
        #Usado no início de toda função que não precisa do slider
        self.desativarSlider()

        #Caso a imagem seja colorida
        if self.im1.tipo == 1:
            imagem, tipo = self.converterParaCinza(self.im1.matriz.copy(), self.im1.tipo)
            imagem, tipo = self.converterBin(self.limiarizarOtsu(self.histograma(imagem),imagem),imagem,tipo)
            #self.im1.matriz, self.im1.tipo = imagem.copy(), tipo
            #self.atualizarImagem('im1')
        elif self.im1.tipo == 2:
            imagem, tipo = self.converterBin(self.limiarizarOtsu(self.histograma(self.im1.matriz.copy()),self.im1.matriz.copy()),self.im1.matriz.copy(), self.im1.tipo)
            #self.im1.matriz, self.im1.tipo = imagem.copy(), tipo
            #self.atualizarImagem('im1')
        elif self.im1.tipo == 3:
            imagem, tipo = self.im1.matriz.copy(), self.im1.tipo


        #Pergunta se é necessário corrigir orientação do texto
        pergunta = MyQuestion("Pergunta:","Deseja corrigir orientação?")
        #Caso não deseje setar todos os pixels
        if pergunta.exec_():
            matriz,angulo = self.transformadaDeHough(imagem)
            self.imRes.matriz = self.novoRotacionar(imagem.copy(),tipo,-angulo)
            self.imRes.tipo = 3
        else:
            self.imRes.matriz = imagem.copy()
            self.imRes.tipo = tipo

        limites = self.segmentacaoLetras(self.imRes.matriz.copy())
        print(limites)

        self.desenharCaixas(limites)

# FUNÇÃO PRINCIPAL
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    exit(app.exec_())
