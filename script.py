### read one file at a time from inputs folder
### extract the data from the file
import os
import numpy as np
import matplotlib.pyplot as plt


folders = ['inputs','input2']

def loss(folder):
    psnr_dict = dict({})
    g_loss_dict = dict({})
    for file in os.listdir(f'{folder}'):
        
        psnr = []
        g_loss = []
        if file.endswith('.txt'):
            ######## read lines from file
            with open(os.path.join(f'{folder}',file), 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if 'step_acc' in lines[i]:
                        step_acc = lines[i].split('step_acc: ')[1].split(',')[0]
                        psnr.append(float(step_acc))
                    if 'G_loss' in lines[i]:
                        g_loss.append(float(lines[i].split('G_loss: ')[1].split(',')[0]))
                        
            psnr_dict[file] = psnr
            g_loss_dict[file] = g_loss
    return psnr_dict, g_loss_dict




array = np.array([])
mean_psnr = dict({})
mean_gloss = dict({})
for folder in folders:
    psnr_dict, g_loss_dict = loss(folder)
    array = np.array([])
    ####### vertically stack the data of values of dict to an array
    for key in psnr_dict.keys():
        array = np.vstack((array, psnr_dict[key])) if array.size else np.array(psnr_dict[key])
    mean = np.mean(array, axis=0)
    mean_psnr[folder] = mean
    
    array = np.array([])
    
    for key in g_loss_dict.keys():
        array = np.vstack((array, g_loss_dict[key])) if array.size else np.array(g_loss_dict[key])
    mean = np.mean(array, axis=0)
    mean_gloss[folder] = mean
    
    array = np.array([])
    
    
    
plt.figure(figsize=(5,5))
plt.title('PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
for key in mean_psnr.keys():
    plt.plot(np.arange(len(mean_psnr[key])), mean_psnr[key], label=key)
plt.legend(mean_psnr.keys()) 
plt.savefig('images/psnr.png')

plt.figure(figsize=(5,5))
plt.title('G_loss')
plt.xlabel('Epoch')
plt.ylabel('G_loss')
for key in mean_gloss.keys():
    plt.plot(np.arange(len(mean_gloss[key])), mean_gloss[key], label=key)
plt.legend(mean_psnr.keys()) 
plt.savefig('images/gloss.png')

