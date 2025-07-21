import pandas as pd
import numpy as np
from io import StringIO
import ace_tools as tools

# Read the provided data into a DataFrame
data = """
Model	Dataset	DL algorithm	accuracy	precision	recall	f1	auc
Baseline CNN	CIC_IOT_DIAD	CNN	0.8728790089	0.9868285987	0.8671480864	0.9231254501	0.8910630639
Optimized CNN	CIC_IOT_DIAD	CNN	0.8732528941	0.9860463454	0.8682860829	0.9234270034	0.8890124459
Quantized + Pruned CNN	CIC_IOT_DIAD	CNN	0.8894822843	0.9645661174	0.9077855742	0.9353148884	0.831406463
Baseline CNN+GRU	CIC_IOT_DIAD	CNN & GRU	0.9149664888	0.9971658797	0.9059658286	0.9493806535	0.943525329
Optimized CNN+GRU	CIC_IOT_DIAD	CNN & GRU	0.9048485072	0.9982188891	0.893489821	0.9429553417	0.9408892976
Quantized + Pruned CNN+GRU	CIC_IOT_DIAD	CNN & GRU	0.9068656413	0.9975024071	0.8964318303	0.9442702824	0.9399718286
Baseline GRU	CIC_IOT_DIAD	GRU	0.9095797714	0.9991015431	0.8980785165	0.94590034	0.946072928
Optimized GRU	CIC_IOT_DIAD	GRU	0.9089566293	0.9991416359	0.8973338368	0.9455050809	0.9458354214
Quantized + Pruned GRU	CIC_IOT_DIAD	GRU	0.9060717121	0.9993199233	0.8938936262	0.9436713679	0.9447124345
Baseline LSTM	CIC_IOT_DIAD	LSTM	0.9062794262	0.9994547629	0.8940089991	0.943795778	0.9452131443
Optimized LSTM	CIC_IOT_DIAD	LSTM	0.9086519821	0.9991821283	0.8969510085	0.9453106433	0.9457788404
Quantized + Pruned LSTM	CIC_IOT_DIAD	LSTM	0.9064317498	0.9991388855	0.8944652465	0.9439090192	0.9444011262
Baseline SimpleRNN	CIC_IOT_DIAD	Simple RNN	0.9087627629	0.9989723948	0.8972656619	0.9453914543	0.9452427393
Optimized SimpleRNN	CIC_IOT_DIAD	Simple RNN	0.906390207	0.9994079996	0.8941768142	0.9438684285	0.9451429568
Quantized + Pruned SimpleRNN	CIC_IOT_DIAD	Simple RNN	0.9066486956	0.9990514445	0.8947903884	0.9440509915	0.9442747689
Baseline Transformer	CIC_IOT_DIAD	Distilled Transformer	0.8940935359	0.9889410969	0.8896248283	0.9366576299	0.9082726191
Optimized Transformer	CIC_IOT_DIAD	Distilled Transformer	0.8918548402	0.9904868593	0.8856392184	0.9351333243	0.9115768324
Quantized Transformer	CIC_IOT_DIAD	Distilled Transformer	0.8951321061	0.9893544496	0.8904376829	0.9372935108	0.9100273783
"""

df = pd.read_csv(StringIO(data), sep="\t")

# Compute summary statistics for each DL algorithm and Dataset combination
summary = df.groupby(['Dataset', 'DL algorithm']).agg({
    'accuracy': ['mean', 'std'],
    'precision': ['mean', 'std'],
    'recall': ['mean', 'std'],
    'f1': ['mean', 'std'],
    'auc': ['mean', 'std']
}).round(4)

# Flatten the multi-level columns
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary.reset_index(inplace=True)

# Display the summary table
tools.display_dataframe_to_user(name="Model Performance Summary", dataframe=summary)
