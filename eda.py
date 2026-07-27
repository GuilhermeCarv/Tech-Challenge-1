import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


class EDA:
    """Classe para Exploratory Data Analysis (EDA) completa de datasets."""
    
    def __init__(self, df: pd.DataFrame, figsize: tuple = (12, 6),colunas: Optional[List[str]] = None):
        """
        Inicializa a classe EDA.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame para análise
        figsize : tuple
            Tamanho padrão das figuras
        """
        self.df = df
        self.figsize = figsize
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize

    def _obter_colunas_numericas(self, df: Optional[pd.DataFrame] = None, colunas: Optional[List[str]] = None) -> List[str]:
        """Retorna colunas numéricas usando pandas.is_numeric_dtype (suporta dtypes 'extension').

        Nota: não filtra colunas que sejam todas NaN — as funções de plotagem já fazem .dropna() antes de desenhar.
        """
        if df is None:
            df = self.df

        from pandas.api.types import is_numeric_dtype

        numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c].dtype)]

        if colunas is not None:
            return [col for col in colunas if col in numeric_cols]

        return numeric_cols

    def _obter_colunas_categoricas(self, df: Optional[pd.DataFrame] = None, colunas: Optional[List[str]] = None,
                                   low_cardinality_threshold: Optional[int] = None) -> List[str]:
        """Retorna colunas categóricas (category, object, string, bool).
        Se low_cardinality_threshold for fornecido, considera colunas numéricas com poucas categorias como categóricas.
        """
        if df is None:
            df = self.df

        from pandas.api.types import (is_categorical_dtype, is_object_dtype,
                                      is_string_dtype, is_bool_dtype, is_numeric_dtype)

        cat_cols = []
        for col in df.columns:
            dtype = df[col].dtype
            if (is_categorical_dtype(dtype) or is_object_dtype(dtype)
                    or is_string_dtype(dtype) or is_bool_dtype(dtype)):
                cat_cols.append(col)
            elif low_cardinality_threshold is not None and is_numeric_dtype(dtype):
                if df[col].nunique(dropna=True) <= low_cardinality_threshold:
                    cat_cols.append(col)

        if colunas is not None:
            return [col for col in colunas if col in cat_cols]

        return cat_cols
    
    def verificar_nulos(self) -> pd.DataFrame:
        """Verifica e exibe informações sobre valores nulos."""
        print("\n" + "="*60)
        print("ANÁLISE DE VALORES NULOS")
        print("="*60)
        
        nulos = pd.DataFrame({
            'Coluna': self.df.columns,
            'Nulos': self.df.isnull().sum().values,
            'Percentual (%)': (self.df.isnull().sum().values / len(self.df) * 100).round(2)
        })
        
        print(nulos.to_string(index=False))
        
        # Visualizar nulos com heatmap
        plt.figure(figsize=self.figsize)
        sns.heatmap(self.df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
        plt.title('Mapa de Valores Nulos')
        plt.tight_layout()
        plt.show()
        
        return nulos
    
    def info_geral(self) -> None:
        """Exibe informações gerais do dataset."""
        print("\n" + "="*60)
        print("INFORMAÇÕES GERAIS DO DATASET")
        print("="*60)
        print(f"Dimensões: {self.df.shape[0]} linhas × {self.df.shape[1]} colunas")
        print(f"\nTipos de dados:\n{self.df.dtypes}")
        print(f"\nMemória utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def estatisticas_descritivas(self, colunas: Optional[List[str]] = None) -> pd.DataFrame:
        """Retorna estatísticas descritivas das colunas numéricas."""
        print("\n" + "="*60)
        print("ESTATÍSTICAS DESCRITIVAS")
        print("="*60)

        if colunas is not None:
            colunas = [col for col in colunas if col in self.df.columns]
            df = self.df[colunas]
        else:
            df = self.df

        col_numericas = self._obter_colunas_numericas(df)
        df_numericos = df[col_numericas]
        if df_numericos.empty:
            print("Nenhuma coluna numérica encontrada para estatísticas descritivas!")
            return pd.DataFrame()

        stats = df_numericos.describe().T
        print(stats)
        return stats
    
    def boxplot_distribuicao(self, colunas: Optional[List[str]] = None) -> None:
        """
        Cria boxplots para visualizar distribuição e outliers.
        
        Parameters:
        -----------
        colunas : list, optional
            Colunas numéricas para análise. Se None, seleciona automaticamente.
        """
        if colunas is not None:
            colunas = self._obter_colunas_numericas(colunas=colunas)
        else:
            colunas = self._obter_colunas_numericas()
        
        n_cols = len(colunas)
        if n_cols == 0:
            print("Nenhuma coluna numérica encontrada!")
            return
        
        fig, axes = plt.subplots(1, min(n_cols, 4), figsize=(15, 5))
        if n_cols == 1:
            axes = [axes]

        for idx, col in enumerate(colunas[:-1]):
            series = self.df[col].dropna()
            if series.empty:
                axes[idx].text(0.5, 0.5, 'Sem valores não-nulos', ha='center', va='center')
                axes[idx].set_title(f'Boxplot - {col} (sem dados)')
                continue
            sns.boxplot(y=series, ax=axes[idx], palette='Set2')
            axes[idx].set_title(f'Boxplot - {col}')

        plt.tight_layout()
        plt.show()
    
    def histograma_distribuicao(self, colunas: Optional[List[str]] = None) -> None:
        """
        Cria histogramas para visualizar distribuição das variáveis.
        
        Parameters:
        -----------
        colunas : list, optional
            Colunas para análise. Se None, seleciona colunas numéricas.
        """
        if colunas is not None:
            colunas = self._obter_colunas_numericas(colunas=colunas)
        else:
            colunas = self._obter_colunas_numericas()
        
        n_cols = len(colunas)
        if n_cols == 0:
            print("Nenhuma coluna numérica encontrada!")
            return
        
        fig, axes = plt.subplots((n_cols + 2) // 3, 3, figsize=(15, 10))
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(colunas):
            series = self.df[col].dropna()
            if series.empty:
                axes[idx].text(0.5, 0.5, 'Sem valores não-nulos', ha='center', va='center')
                axes[idx].set_title(f'Distribuição - {col} (sem dados)')
                continue
            # Não passar `palette` quando não há `hue` para evitar UserWarning
            sns.histplot(x=series, kde=True, ax=axes[idx])
            axes[idx].set_title(f'Distribuição - {col}')

        for idx in range(len(colunas), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()
    
    def analise_correlacao(self, metodo: str = 'pearson', colunas: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analisa correlação entre variáveis numéricas.
        
        Parameters:
        -----------
        metodo : str
            Método de correlação: 'pearson', 'spearman' ou 'kendall'
        colunas : list, optional
            Colunas numéricas para análise
        """
        print(f"\n" + "="*60)
        print(f"MATRIZ DE CORRELAÇÃO ({metodo.upper()})")
        print("="*60)
        
        if colunas is not None:
            colunas = [col for col in colunas if col in self.df.columns]
            df_numericos = self.df[self._obter_colunas_numericas(self.df, colunas)]
        else:
            df_numericos = self.df[self._obter_colunas_numericas()]

        if df_numericos.empty:
            print("Nenhuma coluna numérica encontrada para correlação!")
            return pd.DataFrame()

        correlacao = df_numericos.corr(method=metodo)
        print(correlacao)
        
        # Heatmap de correlação
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlacao, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True, linewidths=1)
        plt.title(f'Matriz de Correlação - {metodo.capitalize()}')
        plt.tight_layout()
        plt.show()
        
        return correlacao
    
    def correlacao_com_target(self, target: str, top_n: int = 10) -> pd.DataFrame:
        """
        Mostra as correlações mais fortes com a variável alvo.
        
        Parameters:
        -----------
        target : str
            Nome da coluna alvo
        top_n : int
            Número de correlações a exibir
        """
        print(f"\n" + "="*60)
        print(f"TOP {top_n} CORRELAÇÕES COM '{target}'")
        print("="*60)
        
        df_numericos = self.df[self._obter_colunas_numericas()]
        correlacao = df_numericos.corr()[target].sort_values(ascending=False)
        print(correlacao.head(top_n))
        
        return correlacao.head(top_n)
    
    def valores_unicos(self) -> None:
        """Exibe contagem de valores únicos por coluna."""
        print("\n" + "="*60)
        print("CONTAGEM DE VALORES ÚNICOS")
        print("="*60)
        
        unicos = pd.DataFrame({
            'Coluna': self.df.columns,
            'Valores Únicos': self.df.nunique().values,
            'Percentual (%)': (self.df.nunique().values / len(self.df) * 100).round(2)
        })
        
        print(unicos.to_string(index=False))
    
    def analise_categoricas(self, max_categorias: int = 10) -> None:
        """
        Analisa distribuição de colunas categóricas.
        
        Parameters:
        -----------
        max_categorias : int
            Limite de categorias para visualização
        """
        colunas_cat = self._obter_colunas_categoricas()
        
        if not colunas_cat:
            print("Nenhuma coluna categórica encontrada!")
            return
        
        for col in colunas_cat[:4]:
            print(f"\n{col}:")
            print(self.df[col].value_counts().head(max_categorias))
            
            plt.figure(figsize=(10, 5))
            self.df[col].value_counts().head(max_categorias).plot(kind='bar', color='skyblue')
            plt.title(f'Distribuição - {col}')
            plt.xlabel(col)
            plt.ylabel('Frequência')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def deteccao_outliers(self, metodo: str = 'iqr', colunas: Optional[List[str]] = None) -> dict:
        """
        Detecta outliers usando IQR ou Z-score.
        
        Parameters:
        -----------
        metodo : str
            'iqr' para Interquartile Range ou 'zscore'
        colunas : list, optional
            Colunas para análise
        """
        if colunas is not None:
            colunas = self._obter_colunas_numericas(colunas=colunas)
        else:
            colunas = self._obter_colunas_numericas()
        
        if not colunas:
            print("Nenhuma coluna numérica encontrada para detecção de outliers!")
            return {}
        
        outliers = {}
        
        print(f"\n" + "="*60)
        print(f"DETECÇÃO DE OUTLIERS (Método: {metodo.upper()})")
        print("="*60)
        
        for col in colunas:
            if metodo == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                limite_inf = Q1 - 1.5 * IQR
                limite_sup = Q3 + 1.5 * IQR
                outliers_mask = (self.df[col] < limite_inf) | (self.df[col] > limite_sup)
            else:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers_mask = z_scores > 3
            
            n_outliers = outliers_mask.sum()
            outliers[col] = n_outliers
            print(f"{col}: {n_outliers} outliers ({n_outliers/len(self.df)*100:.2f}%)")
        
        return outliers
    
    def relatorio_completo(self, colunas: Optional[List[str]] = None,
            metodo_correlacao: str = 'pearson',
            metodo_outliers: str = 'iqr',
            max_categorias: int = 10) -> None:
        """Executa análise completa de EDA.

        Parameters:
        -----------
        colunas : list, optional
            Colunas específicas para análises numéricas
        metodo_correlacao : str
            Método de correlação
        metodo_outliers : str
            Método de detecção de outliers
        max_categorias : int
            Limite de categorias para visualização
        """
        print("\n" + "="*80)
        print("RELATÓRIO COMPLETO DE EDA - EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        self.info_geral()
        self.verificar_nulos()
        self.valores_unicos()
        self.estatisticas_descritivas(colunas)
        self.histograma_distribuicao(colunas)
        self.boxplot_distribuicao(colunas)
        self.analise_categoricas(max_categorias=max_categorias)
        self.analise_correlacao(metodo=metodo_correlacao, colunas=colunas)
        self.deteccao_outliers(metodo=metodo_outliers, colunas=colunas)
        
        print("\n" + "="*80)
        print("FIM DO RELATÓRIO DE EDA")
        print("="*80)


# Exemplo de uso
if __name__ == "__main__":
    # Carregar dados de exemplo
    df = pd.read_csv('seu_arquivo.csv')
    
    # Inicializar e executar EDA
    eda = EDA(df)
    eda.relatorio_completo()
