"""Script de engenharia de features para o dataset Telco Customer Churn.

Este script carrega o conjunto de dados de churn, aplica limpeza e engenharia de
features, e grava um conjunto preparado pronto para modelagem.

Uso:
    python src/feature_engineering.py \
        --input Telco_customer_churn.xlsx \
        --output data/processed/telco_feature_engineered.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


def carregar_dataset(caminho_entrada: Path) -> pd.DataFrame:
    LOGGER.info('Carregando dataset de %s', caminho_entrada)
    if caminho_entrada.suffix.lower() in {'.xlsx', '.xls'}:
        df = pd.read_excel(caminho_entrada)
    else:
        df = pd.read_csv(caminho_entrada)
    LOGGER.info('Dataset carregado com shape %s', df.shape)
    return df


def limpar_cobrancas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Total Charges' in df.columns:
        df['Total Charges'] = (
            df['Total Charges']
            .astype(str)
            .str.replace(' ', '', regex=False)
            .replace(['', 'nan', 'NaN'], pd.NA)
        )
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0.0)
    if 'Monthly Charges' in df.columns:
        df['Monthly Charges'] = pd.to_numeric(df['Monthly Charges'], errors='coerce').fillna(0.0)
    if 'CLTV' in df.columns:
        df['CLTV'] = pd.to_numeric(df['CLTV'], errors='coerce').fillna(0.0)
    if 'Tenure Months' in df.columns:
        df['Tenure Months'] = pd.to_numeric(df['Tenure Months'], errors='coerce').fillna(0).astype(int)
    return df


def binario_sim_nao(serie: pd.Series, valores_sim=None) -> pd.Series:
    valores_sim = valores_sim or {'Yes'}
    return serie.fillna('No').astype(str).str.strip().replace({v: 1 for v in valores_sim}).replace({'No': 0, 'no': 0, 'NO': 0})


def normalizar_coluna_servico(serie: pd.Series) -> pd.Series:
    return serie.fillna('No').replace({'No internet service': 'No', 'No phone service': 'No'})


def criar_faixa_tenure(tenure: pd.Series) -> pd.Series:
    bins = [-1, 6, 12, 24, 48, 72]
    labels = ['0-6', '7-12', '13-24', '25-48', '49+']
    return pd.cut(tenure, bins=bins, labels=labels)


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = limpar_cobrancas(df)
    df = df.copy()

    colunas_para_remover = [
        'CustomerID',
        'Count',
        'Country',
        'State',
        'Zip Code',
        'Lat Long',
        'Latitude',
        'Longitude',
        'Churn Label',
        'Churn Reason',
    ]
    colunas_existentes = [col for col in colunas_para_remover if col in df.columns]
    if colunas_existentes:
        df = df.drop(columns=colunas_existentes)

    df['Churn Value'] = df['Churn Value'].astype(int)
    df['churn_target'] = df['Churn Value']

    df['is_senior'] = binario_sim_nao(df['Senior Citizen'], valores_sim={'Yes'})
    df['has_partner'] = binario_sim_nao(df['Partner'], valores_sim={'Yes'})
    df['has_dependents'] = binario_sim_nao(df['Dependents'], valores_sim={'Yes'})
    df['is_paperless_billing'] = binario_sim_nao(df['Paperless Billing'], valores_sim={'Yes'})
    df['payment_method_auto'] = df['Payment Method'].astype(str).str.contains('automatic', case=False, na=False).astype(int)
    df['is_month_to_month'] = df['Contract'].astype(str).str.contains('Month-to-month', case=False, na=False).astype(int)
    df['contract_term'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2}).fillna(-1).astype(int)
    df['contract_months'] = df['Contract'].map({'Month-to-month': 1, 'One year': 12, 'Two year': 24}).fillna(1).astype(int)

    for coluna in [
        'Online Security',
        'Online Backup',
        'Device Protection',
        'Tech Support',
        'Streaming TV',
        'Streaming Movies',
        'Multiple Lines',
    ]:
        if coluna in df.columns:
            df[coluna] = normalizar_coluna_servico(df[coluna])

    df['has_phone_service'] = binario_sim_nao(df['Phone Service'], valores_sim={'Yes'})
    df['has_multiple_lines'] = binario_sim_nao(df['Multiple Lines'], valores_sim={'Yes'})
    df['has_internet_service'] = df['Internet Service'].fillna('No').replace({'No': 0, 'DSL': 1, 'Fiber optic': 1}).astype(int)
    df['has_dsl'] = (df['Internet Service'] == 'DSL').astype(int)
    df['has_fiber_optic'] = (df['Internet Service'] == 'Fiber optic').astype(int)
    df['offers_online_security'] = binario_sim_nao(df['Online Security'], valores_sim={'Yes'})
    df['offers_online_backup'] = binario_sim_nao(df['Online Backup'], valores_sim={'Yes'})
    df['offers_device_protection'] = binario_sim_nao(df['Device Protection'], valores_sim={'Yes'})
    df['offers_tech_support'] = binario_sim_nao(df['Tech Support'], valores_sim={'Yes'})
    df['offers_streaming_tv'] = binario_sim_nao(df['Streaming TV'], valores_sim={'Yes'})
    df['offers_streaming_movies'] = binario_sim_nao(df['Streaming Movies'], valores_sim={'Yes'})

    df['service_count'] = (
        df[
            [
                'has_phone_service',
                'has_multiple_lines',
                'has_internet_service',
                'offers_online_security',
                'offers_online_backup',
                'offers_device_protection',
                'offers_tech_support',
                'offers_streaming_tv',
                'offers_streaming_movies',
            ]
        ].sum(axis=1)
    )

    df['internet_only_customer'] = ((df['has_internet_service'] == 1) & (df['has_phone_service'] == 0)).astype(int)
    df['all_services_with_internet'] = ((df['has_internet_service'] == 1) & (df['service_count'] >= 5)).astype(int)
    df['high_value_customer'] = (df['CLTV'] > df['CLTV'].median()).astype(int)
    df['tenure_bucket'] = criar_faixa_tenure(df['Tenure Months'])
    df['avg_charge_per_month'] = df['Total Charges'] / df['Tenure Months'].replace(0, 1)
    df['charge_ratio'] = df['Total Charges'] / df['Monthly Charges'].replace(0, 1)
    df['avg_monthly_charge_delta'] = df['Monthly Charges'] - df['avg_charge_per_month']

    colunas_feature = [
        'Churn Value',
        'churn_target',
        'is_senior',
        'has_partner',
        'has_dependents',
        'is_paperless_billing',
        'payment_method_auto',
        'is_month_to_month',
        'contract_term',
        'contract_months',
        'has_phone_service',
        'has_multiple_lines',
        'has_internet_service',
        'has_dsl',
        'has_fiber_optic',
        'offers_online_security',
        'offers_online_backup',
        'offers_device_protection',
        'offers_tech_support',
        'offers_streaming_tv',
        'offers_streaming_movies',
        'service_count',
        'internet_only_customer',
        'all_services_with_internet',
        'high_value_customer',
        'tenure_bucket',
        'avg_charge_per_month',
        'charge_ratio',
        'avg_monthly_charge_delta',
    ]

    LOGGER.info('Engenharia de features concluída com %d colunas', len(df.columns))
    LOGGER.info('Features geradas: %s', [c for c in colunas_feature if c in df.columns])
    return df


def gravar_dataset(df: pd.DataFrame, caminho_saida: Path) -> None:
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info('Gravando dataset engenheirado em %s', caminho_saida)
    if caminho_saida.suffix.lower() in {'.parquet'}:
        df.to_parquet(caminho_saida, index=False)
    else:
        df.to_csv(caminho_saida, index=False)
    LOGGER.info('Dataset salvo com shape %s', df.shape)


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script de engenharia de features para churn')
    parser.add_argument('--input', type=Path, default=Path('Telco_customer_churn.xlsx'), help='Caminho para o dataset bruto')
    parser.add_argument('--output', type=Path, default=Path('data/processed/telco_feature_engineered.csv'), help='Caminho para salvar o dataset processado')
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parsear_argumentos()
    df = carregar_dataset(args.input)
    df_engineered = preparar_features(df)
    gravar_dataset(df_engineered, args.output)


if __name__ == '__main__':
    main()
