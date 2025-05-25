**Relatório do Projeto: Classificação de Imagens MNIST com PyTorch**

**1\. Introdução**

Este projeto tem como objetivo demonstrar a implementação de um modelo de rede neural para a tarefa de classificação de imagens utilizando a biblioteca PyTorch. O conjunto de dados escolhido para este fim é o MNIST, um clássico em aprendizado de máquina, composto por imagens de dígitos manuscritos de 0 a 9\. O PyTorch foi selecionado devido à sua flexibilidade, facilidade de uso e eficiência na construção e treinamento de modelos de aprendizado profundo.

**2\. Metodologia**

A metodologia adotada neste projeto envolve as seguintes etapas:

* **Importação de Bibliotecas:** As bibliotecas PyTorch, Matplotlib e NumPy, necessárias para o desenvolvimento do projeto foram importadas.  
* **Transformação de Dados:** As imagens do conjunto de dados MNIST foram transformadas para o formato adequado para entrada no modelo.  
* **Carregamento dos Dados MNIST:** Os conjuntos de dados de treinamento e teste do MNIST foram carregados e preparados para o treinamento e avaliação do modelo.  
* **Definição da Arquitetura da Rede Neural:** A arquitetura do modelo de rede neural foi definida utilizando as camadas e funções de ativação apropriadas.  
* **Instanciação do Modelo:** O modelo de rede neural definido foi instanciado.  
* **Definição da Função de Perda e Otimizador:** A função de perda e o otimizador foram definidos para guiar o processo de treinamento do modelo.  
* **Treinamento do Modelo:** O modelo foi treinado utilizando o conjunto de dados de treinamento.  
* **Avaliação do Modelo:** O desempenho do modelo treinado foi avaliado utilizando o conjunto de dados de teste.

**3\. Detalhamento das Etapas e Uso do PyTorch**

**3.1. Importação de Bibliotecas**

As seguintes bibliotecas foram importadas:

* `torch`: Biblioteca base do PyTorch para operações com tensores e construção de redes neurais.  
* `torch.nn as nn`: Módulo do PyTorch para definição de arquiteturas de redes neurais.  
* `torch.optim as optim`: Módulo do PyTorch para algoritmos de otimização, como o Adam.  
* `torchvision`: Pacote do PyTorch que fornece acesso a conjuntos de dados populares (como o MNIST), arquiteturas de modelos pré-treinados e transformações de imagens.  
* `torchvision.transforms as transforms`: Módulo do PyTorch para realizar transformações em imagens, como normalização e conversão para tensores.  
* `matplotlib.pyplot as plt`: Biblioteca para visualização de dados (não utilizada diretamente no código, mas útil para análise exploratória).  
* `numpy as np`: Biblioteca para operações numéricas (não essencial para este código, mas frequentemente usada em conjunto com PyTorch).

**3.2. Transformação de Dados**

As imagens do conjunto de dados MNIST foram transformadas utilizando a classe `transforms.Compose` do PyTorch, que permite a aplicação de uma sequência de transformações. As transformações aplicadas foram:

* `transforms.ToTensor()`: Converte as imagens para tensores PyTorch. Tensores são a estrutura de dados fundamental do PyTorch, similares a arrays multidimensionais.  
* `transforms.Normalize((0.5,), (0.5,))`: Normaliza os tensores das imagens para que seus valores fiquem em uma faixa específica (neste caso, com média 0.5 e desvio padrão 0.5). A normalização é importante para acelerar o treinamento e melhorar a estabilidade do modelo.

**3.3. Carregamento dos Dados MNIST**

Os conjuntos de dados de treinamento e teste do MNIST foram carregados utilizando a classe `torchvision.datasets.MNIST`. Os parâmetros utilizados foram:

* `root='./data'`: Especifica o diretório onde os dados serão armazenados.  
* `train=True/False`: Indica se o conjunto de dados a ser carregado é o de treinamento ou o de teste.  
* `download=True`: Se os dados não estiverem presentes no diretório, eles serão baixados automaticamente.  
* `transform=transform`: Aplica as transformações definidas anteriormente aos dados.

Os dados carregados foram então encapsulados em objetos `torch.utils.data.DataLoader`. O `DataLoader` facilita a iteração sobre o conjunto de dados em *batches* (lotes), o que é essencial para o treinamento eficiente de modelos de aprendizado profundo. Os parâmetros utilizados no `DataLoader` foram:

* `batch_size=64`: Define o número de amostras em cada lote.  
* `shuffle=True/False`: Embaralha os dados (apenas no conjunto de treinamento) para garantir que o modelo não aprenda padrões espúrios devido à ordem dos dados.

**3.4. Definição da Arquitetura da Rede Neural**

A arquitetura da rede neural foi definida como uma classe chamada `Net` que herda de `nn.Module`, a classe base para todos os módulos de rede neural no PyTorch. A rede neural é composta por três camadas totalmente conectadas (lineares):

* `self.fc1 = nn.Linear(28 * 28, 128)`: A primeira camada recebe como entrada vetores de tamanho 28\*28 (o tamanho de cada imagem MNIST achatada) e produz vetores de tamanho 128\.  
* `self.fc2 = nn.Linear(128, 64)`: A segunda camada mapeia vetores de 128 para 64\.  
* `self.fc3 = nn.Linear(64, 10)`: A camada de saída, que mapeia vetores de 64 para 10 (o número de classes, correspondentes aos dígitos de 0 a 9).

A função `forward` define o fluxo de dados através da rede. As operações realizadas são:

* `x = x.view(-1, 28 * 28)`: Achata o tensor de entrada (que originalmente tem a forma de uma imagem 2D) em um vetor 1D.  
* `x = torch.relu(self.fc1(x))`: Aplica a função de ativação ReLU à saída da primeira camada linear. ReLU introduz não-linearidade, permitindo que a rede aprenda relações mais complexas.  
* `x = torch.relu(self.fc2(x))`: Aplica ReLU à saída da segunda camada linear.  
* `x = self.fc3(x)`: A saída da terceira camada.

**3.5. Instanciação do Modelo**

Uma instância da rede neural (`net`) foi criada a partir da classe `Net`.

**3.6. Definição da Função de Perda e Otimizador**

* `criterion = nn.CrossEntropyLoss()`: A função de perda CrossEntropyLoss foi escolhida, adequada para problemas de classificação multiclasse. Ela mede a diferença entre as previsões do modelo e os rótulos verdadeiros.  
* `optimizer = optim.Adam(net.parameters(), lr=0.001)`: O otimizador Adam foi selecionado para treinar a rede. Adam é um algoritmo de otimização que ajusta os parâmetros da rede (pesos) para minimizar a função de perda. O `lr=0.001` define a taxa de aprendizado, um hiperparâmetro que controla o tamanho dos passos que o otimizador dá para atualizar os parâmetros.

**3.7. Treinamento do Modelo**

O modelo foi treinado por 5 épocas. Uma época representa uma passagem completa pelo conjunto de treinamento. O processo de treinamento envolveu:

* Iteração sobre os lotes de dados de treinamento.  
* `optimizer.zero_grad()`: Zera os gradientes antes de cada passagem para evitar o acúmulo de gradientes de iterações anteriores.  
* `outputs = net(inputs)`: Passa as imagens do lote pela rede para obter as previsões.  
* `loss = criterion(outputs, labels)`: Calcula a perda comparando as previsões com os rótulos verdadeiros.  
* `loss.backward()`: Realiza a retropropagação para calcular os gradientes da perda em relação aos parâmetros da rede.  
* `optimizer.step()`: Atualiza os parâmetros da rede usando o otimizador e os gradientes calculados.  
* Monitoramento e impressão da perda a cada 1000 iterações para acompanhar o progresso do treinamento.

**3.8. Avaliação do Modelo**

Após o treinamento, o modelo foi avaliado no conjunto de teste para medir seu desempenho em dados não vistos. A avaliação envolveu:

* `with torch.no_grad()`: Desativa o cálculo de gradientes durante a avaliação para economizar memória e computação.  
* Iteração sobre os lotes do conjunto de teste.  
* Obtenção das previsões do modelo para as imagens do lote.  
* Cálculo da acurácia, que é a porcentagem de amostras classificadas corretamente.

**4\. Resultados**

O modelo treinado atingiu uma acurácia de aproximadamente 96% no conjunto de teste. Isso demonstra que o modelo é capaz de classificar com sucesso a maioria dos dígitos escritos à mão do MNIST.

**5\. Conclusão**

Este projeto demonstrou a implementação de um classificador de imagens utilizando PyTorch. A alta acurácia alcançada no conjunto de teste indica que o modelo aprendeu com sucesso a classificar os dígitos do MNIST. Este projeto serve como uma base para explorar tarefas mais complexas de visão computacional e aprendizado profundo utilizando o PyTorch.
