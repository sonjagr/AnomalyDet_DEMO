import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
patches = [[231603,199], [142,616]]
wholes = [[3022,206], [11,286]]

df_cm = pd.DataFrame(patches, index = ["Normal", "Anomalous"], columns = ["Normal", "Anomalous"])

plt.figure(figsize = (10,7))
sn.set(font_scale=2)
plt.title("Confusion matrix for patches")
s = sn.heatmap(df_cm, cmap="YlGnBu", annot=True, fmt=".0f",norm=LogNorm())
s.set_xlabel('Predicted')
s.set_ylabel('Ground truth')
#plt.tight_layout()
plt.savefig("C:/Users/sgroenro/cernbox/Pictures/cm_patch.png", dpi = 600)
plt.show()

df_cmn = pd.DataFrame(wholes, index = ["Normal", "Anomalous"],
                  columns = ["Normal", "Anomalous"])
sum = 3022 + 206 + 11 +286
sum = 231603  + 199 + 142 + 616
print(sum)
wholesn = [[3022/sum,206/sum], [11/sum,286/sum]]
patchesn = [[231603/(231603+199),199/(231603+199)], [142/(142+616),616/(616+142)]]
df_cmn = pd.DataFrame(patchesn, index = ["Normal", "Anomalous"],
                  columns = ["Normal", "Anomalous"])
plt.figure(figsize = (10,7))
sn.set(font_scale=2)
plt.title("Confusion matrix normalized over the ""\n" " ground truths for patches (N=3525)")
s = sn.heatmap(df_cmn, cmap="YlGnBu", annot=True, fmt=".3f",norm=LogNorm())
s.set_xlabel('Predicted')
s.set_ylabel('Ground truth')
#plt.tight_layout()
plt.savefig("C:/Users/sgroenro/cernbox/Pictures/cm_patchn.png", dpi = 600)
plt.show()

patchesn = [[231603/(231603+142),199/(616+199)], [142/(231603+142),616/(616+199)]]
df_cmn = pd.DataFrame(patchesn, index = ["Normal", "Anomalous"],
                  columns = ["Normal", "Anomalous"])
plt.figure(figsize = (10,7))
sn.set(font_scale=2)
plt.title(r"Confusion matrix normalized over the ""\n" "predictions for patches (N=3525)")
s = sn.heatmap(df_cmn, cmap="YlGnBu", annot=True, fmt=".3f",norm=LogNorm())
s.set_xlabel('Predicted')
s.set_ylabel('Ground truth')
#plt.tight_layout()
plt.savefig("C:/Users/sgroenro/cernbox/Pictures/cm_patchn2.png", dpi = 600)
plt.show()