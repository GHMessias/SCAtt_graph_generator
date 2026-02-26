'''
Arquivo responsável por implementar a metodologia descrita em Clauset 2009 e Broido 2019. Aqui vamos responder duas perguntas principais:

1. Como identificar uma powerlaw ao se deparar com uma?
2. Quão forte é a evidência estatística de que essa rede é realmente regida por uma lei de potência?

Receita para analizar dados distribuidos em lei de potência

1. Estimar os parâmetros x_min e alpha a partir dos métodos descritos em clauset 2009:
    - Estimar x_min a partir do teste de kolmogorov-Smirnoff: D = max_{x => x_min} |S(x) - P(x)|
    - Estimar alpha usando Maximum Likelihood Estimators: \hat{\alpha} = 1 + n \left[ \sum^{n}_{i = 1} \ln \frac{x_i}{x_min} \right]^{-1}
esses valores podem ser calculados usando fit.power_law.alpha, fit.power_law.xmin

2. Aplicar o goodness-of-fit test entre os dados e a lei de potência, usando também os métodos em clauset 2009. Essa parte consiste em criar várias distribuições de lei de potência e comparar elas com os dados reais. Caso o p-valor obtido seja maior que 0.1 a hipótese é plausível, caso contrario é rejeitada.

3. Comparar com hipoteses alternativas via likelihood ratio test. Para cada alternativa, se o likelihood ratio é significantemente diferente de 0, e o valor de p é menor que 0.1, então existe uma distribuição favorecida nesse caso.
'''

import powerlaw

class PowerLawAnalysis:
    '''
    Docstring for PowerLawAnalysis

    Clase base para realizar os procedimentos de verificação de distribuição de potência. 

    1. Pega uma distribuição de lei de potência
    2. Coloca as informações em uma variavel tipo dicionario
    3. Aplica o teste de Broido 2019, só que a nossa versão, mais simplificada para os objetivos do trabalho.
    '''
    def __init__(self, degree_distribution, lrt_range = 1000):
        self.degree_distribution = degree_distribution
        self.distribution_values = dict()
        self.lrt_range = lrt_range

        
    
    def generate_values(self):
        self.fit = powerlaw.Fit(self.degree_distribution, discrete=True)
        self.alpha = self.fit.power_law.alpha
        self.n_tail = self.fit.n_tail
        self.D = self.fit.power_law.D
        # self.p_value = self.fit.power_law.D
        count = 0
        for _ in range(self.lrt_range):
            synth = self.fit.power_law.generate_random(int(self.fit.n_tail))
            fit_s = powerlaw.Fit(synth, discrete = True, xmin = self.fit.xmin)

            if fit_s.power_law.D >= self.D:
                count += 1

        self.p_value = count / self.lrt_range
        self.available_distributions = list(self.fit.supported_distributions.keys())
        # print(self.available_distributions)
        # print(self.alpha, self.n_tail, self.p_value)

        return

    def evaluate_distribution(self, n_tail_cut = 50):
        '''
        Avalia todas as distribuições possíveis, verifica se ela se enquadra em
        Not-Scale Free: lei de potência é rejeitada self.p_value < 0.1 ou existe outra distribuição que favorece em detrimento da power-law
        Weak Scale Free Evidence: self.p_value é menor que 0.1, nenhuma outra se favorece em detrimento da powerlaw
        Strong Scale Free Evidence: self.p_value é menor que 0.1, xmin é maior que 50, valor de alpha se encontra entre 2 e 3

        valores possíveis de comparação:
        powerlaw, lognormal, exponential, stretched_exponential, truncated_power_law
        
        :param self: Description
        '''

        self.generate_values()

        self.distribution_class = None
        self.compare_results = {}

        # Se power-law é rejeitada (p pequeno), já classifica como Not Scale-Free
        if self.p_value < 0.1:
            self.distribution_class = "Not Scale-Free"
            return self.distribution_class

        # Comparações com outras distribuições
        for distribution in self.available_distributions:
            if distribution == "power_law":
                continue

            _p, _R = self.fit.distribution_compare("power_law", distribution)
            self.compare_results[distribution] = {"p": _p, "R": _R}

        # Se alguma alternativa for favorecida (p > 0.1 e R < 0), então não é scale-free
        for distribution, res in self.compare_results.items():
            if res["p"] > 0.1 and res["R"] < 0:
                self.distribution_class = "Not Scale-Free"
                return self.distribution_class

        # Se alpha está fora do intervalo "ideal", é Weak (mesmo que power-law não tenha sido rejeitada)
        if not (2 <= self.alpha <= 3):
            self.distribution_class = "Weak Scale-Free"
            return self.distribution_class

        # Se alpha está ok, decide Strong vs Weak pelo tamanho da cauda
        if self.n_tail > n_tail_cut:
            self.distribution_class = "Strong Scale-Free"
        else:
            self.distribution_class = "Weak Scale-Free"

        return self.distribution_class
        

    def summary(self):
        return {
            "alpha" : self.alpha,
            "n_tail" : self.n_tail,
            "D" : self.D,
            "distribution_class" : self.distribution_class
        }