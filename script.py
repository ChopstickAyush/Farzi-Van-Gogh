### read one file at a time from inputs folder
### extract the data from the file
import os
import matplotlib.pyplot as plt
import numpy as np

dir = 'accuracies/accuracies_oil'

psnr_dict = dict({})
g_loss_dict = dict({})
for file in os.listdir(f'{dir}'):
    psnr = []
    g_loss = []
    if file.endswith('.txt'):
        ######## read lines from file
        with open(os.path.join(f'{dir}',file), 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if 'step_acc' in lines[i]:
                    step_acc = lines[i].split('step_acc: ')[1].split(',')[0]
                    psnr.append(float(step_acc))
                if 'G_loss' in lines[i]:
                    g_loss.append(float(lines[i].split('G_loss: ')[1].split(',')[0]))
                    
        psnr_dict[file] = psnr
        g_loss_dict[file] = g_loss
        
##### plot the data in a graph


plt.figure(figsize=(5,5))
plt.title('PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
for key in psnr_dict.keys():
    plt.plot(np.arange(len(psnr_dict[key])), psnr_dict[key], label=key)
plt.legend(psnr_dict.keys()) 
plt.savefig('plots/psnr_oil.png')

plt.figure(figsize=(5,5))
plt.title('G_loss')
plt.xlabel('Epoch')
plt.ylabel('G_loss')
for key in g_loss_dict.keys():
    plt.plot(np.arange(len(g_loss_dict[key])), g_loss_dict[key], label=key)
plt.legend(g_loss_dict.keys()) 
plt.savefig('plots/g_loss_oil.png')

###### print mean psnr and g_loss
avg_psnr = []
avg_g_loss = []
for key in psnr_dict.keys():
    avg_psnr.append(psnr_dict[key][-1])
    
for key in g_loss_dict.keys():
    avg_g_loss.append(g_loss_dict[key][-1])
    
print ('mean psnr: ', np.mean(avg_psnr))
print ('mean g_loss: ', np.mean(avg_g_loss))