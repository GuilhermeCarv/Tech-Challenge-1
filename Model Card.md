# ML Canvas — Projeto de Previsão de Churn Telco

## 1. Problema de negócio
- Reduzir a taxa de churn de clientes da Telco.
- Identificar clientes com maior risco de cancelamento antes que ocorram.
- Priorização de ações de retenção para maximizar impacto financeiro.

## 2. Stakeholders
- Product Owner / Liderança Comercial
- Customer Success / Retenção
- Equipe de Engenharia de Dados
- Data Science / ML
- Operações / Atendimento

## 3. Usuários finais
- Equipe de retenção
- Gestão de relacionamento com cliente
- Analistas de negócio
- Operações comerciais

## 4. Decisão de negócio
- Ações de retenção devem ser direcionadas para clientes com maior propensão de churn.
- A decisão final deve combinar modelo + revisão humana e contexto de negócio.

## 5. Métricas de negócio
- Taxa de churn geral
- Receita preservada (clientes que ficaram)
- Número de clientes contatados preventivamente
- Custo por ação de retenção
- Retorno sobre investimento (ROI)
- Aumento de retenção por segmento

## 6. Métricas de ML
- F1: foco principal
- Recall: 
- Precision
- Accuracy
- ROC AUC

## 7. SLOs (Service Level Objectives)
- Recall mínimo: >= 0.60
- F1 mínimo: >= 0.50
- ROC AUC mínimo: >= 0.84
- Precision alvo: >= 0.50
- Drift monitorado: alertar se mudança de distribuição de features > 10–15%

## 9. Restrições / regras de negócio
- Não usar modelo como único critério para tomada de decisão.
- Revisar impacto por segmento para evitar viés e discriminação.

## 10. Riscos principais
- Falso negativo: cliente evade sem ser identificado
- Falso positivo: esforço de retenção em cliente que não evadiria
- Drift de dados e mudança de comportamento do cliente

## 11. KPIs de sucesso
- Redução percentual da taxa de churn
- Aumento da taxa de retenção em clientes de risco
- Melhor eficiência de campanhas de retenção

## 12. Critérios de sucesso
- O modelo deve identificar corretamente a maior parte dos clientes que evadiriam.
- Deve ter métricas estáveis e monitoramento contínuo.