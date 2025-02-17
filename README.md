## 🚀 Progression du Modèle  

Voici notre progression tout au long du projet :  


**AUC Score**

| **Feature Engineering** | **XGBoost** | **Cat Boost** | **LGBM** |
|-------------------------|-------------|--------------|--------------|
| Baseline model | 0.9446614696303043 | 0.9570346773659776 | 0.0 |
| Baseline + ACP | 0. | 0.| 0 |
| Baseline + Frequency Encoding | 0.9427600333410997 | 0.9277747528642826 | 0. | 
| Baseline + Target Encoding | 0.9999696469007457 | 0.9999763768410286 | 0. |


**XGBoost**

| **Feature Engineering** | **AUC** | **F1** | **Recall** | **Precision** |  
|-------------------------|-------------|--------------|--------------|--------------|
| Baseline model | 0.9446614696303043 | 0.0 | 0.0 | 0.0 |  
| Baseline + ACP | 0. | 0.| 0 | 0.0 |  
| Baseline + Frequency Encoding | 0.9427600333410997 | 0. | 0. | 0. |  
| Baseline + Target Encoding | 0.9999696469007457 | 0.9833 | 0.9833 | 0.9833 |  


**CatBoost**

| **Feature Engineering** | **AUC** | **F1** | **Recall** | **Precision** |  
|-------------------------|-------------|--------------|--------------|--------------|
| Baseline model | 0.9570346773659776 | 0 | 0 | 0.0 |  
| Baseline + ACP | 0 | 0. | 0. | 0. |  
| Baseline + Frequency Encoding | 0.9277747528642826 | 0. | 0. | 0. |  
| Baseline + Target Encoding | 0.9999763768410286 | 0.9848 | 0.9829 | 0.9868 |  



**LGBM**

| **Feature Engineering** | **AUC** | **F1** | **Recall** | **Precision** |  
|-------------------------|-------------|--------------|--------------|--------------|
| Baseline model) | 0. | 0.0 | 0.0 | 0.0 |  
| Baseline + ACP | 0 | 0.| 0 | 0.0 |  
| Baseline + Frequency Encoding | 0. | 0. | 0. | 0. |  
| Baseline + Target Encoding | 0. | 0. | 0. | 0. |  


