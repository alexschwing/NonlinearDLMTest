# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:16:39 2015

@author: yangsong
"""

import re
import matplotlib.pylab as pl
import numpy as np

def getinfo(name):
    #fin = open('log-n-test-0.1-0.01.txt','r')
    fin = open(name,'r')
    txt = fin.read()
    pap = re.compile(r'Average Precision:\s([0-9]+\.[0-9]+)',re.MULTILINE)
    pnorm = re.compile(r'Norm of weights:\s([0-9]+\.[0-9]+)',re.MULTILINE)
    ap = re.findall(pap,txt)
    ap = [float(x) for x in ap]
    norm = re.findall(pnorm,txt)
    norm = [float(x) for x in norm]
    return ap, norm
    
def plot(ap, norm):
    n = len(ap)
    x = np.r_[0:n]
    
    pl.subplot(2,1,1)
    pl.plot(x,ap)
    pl.ylabel('mAP on train')
    
    pl.subplot(2,1,2)
    padding = len(x) - len(norm)
    norm = np.append(norm, [-1]*padding)
    pl.plot(x,norm)
    pl.ylabel('Norm of grads')
    pl.xlabel('# of iteration')    
    fig = pl.gcf()
    fig.set_size_inches(10,20)    
    
def positive():        
    alphas = [1,0.3,0.1,0.03,0.01,0.003,0.001]
    betas = [1,0.1,0.01,0.001]
    epsilons = [10,1,0.1,0.01]
    for alpha in alphas:
        for beta in betas:
            pl.figure()            
            tags = []            
            for epsilon in epsilons:
                name = "10000p-"+str(alpha)+"-"+str(beta)+"-"+str(epsilon)+".txt"
                tags.append(str(alpha)+"-"+str(beta)+"-"+str(epsilon))
                ap,norm = getinfo(name)
                plot(ap,norm)
            pl.legend(tags)
            name = 'Positive'+'-alpha='+str(alpha)+'-beta='+str(beta)+'.png'
            pl.savefig(name,bbox_inches='tight')
            pl.close()
            
    for alpha in alphas:
        for epsilon in epsilons:
            pl.figure()
            tags = []            
            for beta in betas:
                name = "10000p-"+str(alpha)+"-"+str(beta)+"-"+str(epsilon)+".txt"
                tags.append(str(alpha)+"-"+str(beta)+"-"+str(epsilon))
                ap, norm = getinfo(name)
                plot(ap,norm)
            pl.legend(tags)
            name = 'Positive'+'-alpha='+str(alpha)+'-epsilon='+str(epsilon)+'.png'
            pl.savefig(name,bbox_inches='tight')
            pl.close()
    
    for beta in betas:
        for epsilon in epsilons:
            pl.figure()
            tags = []            
            for alpha in alphas:
                name = "10000p-"+str(alpha)+"-"+str(beta)+"-"+str(epsilon)+".txt"
                tags.append(str(alpha)+"-"+str(beta)+"-"+str(epsilon))
                ap, norm = getinfo(name)
                plot(ap,norm)
            pl.legend(tags)
            name = 'Positive'+'-beta='+str(beta)+'-epsilon='+str(epsilon)+'.png'
            pl.savefig(name,bbox_inches='tight')
            pl.close()

def negative():
    alphas = [1,0.3,0.1,0.03,0.01,0.003,0.001]
    betas = [1,0.1,0.01,0.001]
    epsilons = [10,1,0.1,0.01]
    for alpha in alphas:
        for beta in betas:
            pl.figure()
            tags = []            
            for epsilon in epsilons:
                name = "10000n-"+str(alpha)+"-"+str(beta)+"-"+str(epsilon)+".txt"
                tags.append(str(alpha)+"-"+str(beta)+"-"+str(epsilon))
                ap,norm = getinfo(name)
                plot(ap,norm)
            pl.legend(tags)
            name = 'Negative'+'-alpha='+str(alpha)+'-beta='+str(beta)+'.png'
            pl.savefig(name,bbox_inches='tight')
            pl.close()
            
    for alpha in alphas:
        for epsilon in epsilons:
            pl.figure()
            tags = []            
            for beta in betas:
                name = "10000n-"+str(alpha)+"-"+str(beta)+"-"+str(epsilon)+".txt"
                tags.append(str(alpha)+"-"+str(beta)+"-"+str(epsilon))
                ap,norm = getinfo(name)
                plot(ap,norm)
            pl.legend(tags)
            name = 'Negative'+'-alpha='+str(alpha)+'-epsilon='+str(epsilon)+'.png'
            pl.savefig(name,bbox_inches='tight')
            pl.close()
    
    for beta in betas:
        for epsilon in epsilons:
            pl.figure()
            tags = []            
            for alpha in alphas:
                name = "10000n-"+str(alpha)+"-"+str(beta)+"-"+str(epsilon)+".txt"
                tags.append(str(alpha)+"-"+str(beta)+"-"+str(epsilon))
                ap,norm = getinfo(name)
                plot(ap,norm)
            pl.legend(tags)
            name = 'Negative'+'-beta='+str(beta)+'-epsilon='+str(epsilon)+'.png'
            pl.savefig(name,bbox_inches='tight')
            pl.close()

def per():
    alphas = [1,0.3,0.1,0.03,0.01,0.003,0.001]
    betas = [1,0.1,0.01,0.001]
    for alpha in alphas:
        pl.figure()
        tags = []
        for beta in betas:            
            name = "10000per-"+str(alpha)+"-"+str(beta)+".txt"
            tags.append(str(alpha)+"-"+str(beta))
            ap,norm = getinfo(name)
            plot(ap,norm)
        pl.legend(tags)
        name = 'Perceptron'+'-alpha='+str(alpha)+'.png'
        pl.savefig(name,bbox_inches='tight')
        pl.close()
            
    for beta in betas:
        pl.figure()
        tags = []
        for alpha in alphas:            
            name = "10000per-"+str(alpha)+"-"+str(beta)+".txt"
            tags.append(str(alpha)+"-"+str(beta))
            ap,norm = getinfo(name)
            plot(ap,norm)
        pl.legend(tags)
        name = 'Perceptron'+'-beta='+str(beta)+'.png'
        pl.savefig(name,bbox_inches='tight')
        pl.close()
        
def svm():
    alphas = [1,0.3,0.1,0.03,0.01,0.003,0.001]    
    betas = [1,0.1,0.01,0.001]
    for alpha in alphas:
        pl.figure()
        tags = []
        for beta in betas:            
            name = "10000svm-"+str(alpha)+"-"+str(beta)+".txt"
            tags.append(str(alpha)+"-"+str(beta))
            ap,norm = getinfo(name)
            plot(ap,norm)
        pl.legend(tags)
        name = 'APSVM'+'-alpha='+str(alpha)+'.png'
        pl.savefig(name,bbox_inches='tight')
        pl.close()
            
    for beta in betas:
        pl.figure()
        tags = []
        for alpha in alphas:            
            name = "10000svm-"+str(alpha)+"-"+str(beta)+".txt"
            tags.append(str(alpha)+"-"+str(beta))
            ap,norm = getinfo(name)
            plot(ap,norm)
        pl.legend(tags)
        name = 'APSVM'+'-beta='+str(beta)+'.png'
        pl.savefig(name,bbox_inches='tight')
        pl.close()
        
def combine():
    alphas = [1,0.3,0.1,0.03,0.01,0.003,0.001]
    betas = [1,0.1,0.01,0.001]
    epsilons = [10,1,0.1,0.01]
    for alpha in alphas:
        for beta in betas:            
            for epsilon in epsilons:
                names = ["10000p-"+str(alpha)+"-"+str(beta)+"-"+str(epsilon)+".txt",
                        "10000n-"+str(alpha)+"-"+str(beta)+"-"+str(epsilon)+".txt",
                        "10000svm-"+str(alpha)+"-"+str(beta)+".txt",
                        "10000per-"+str(alpha)+"-"+str(beta)+".txt",
                        ]
                pl.figure()
                for name in names:
                    ap,norm = getinfo(name)
                    plot(ap,norm)
                pl.legend(['P','N','SVM','PER'])
                name = "combine " + str(alpha)+"-"+str(beta)+"-"+str(epsilon)+'.png'
                pl.savefig(name,bbox_inches='tight')
                pl.close()                          

def bestAlphaPlot():
    alphas = [1,0.3,0.1,0.03,0.01,0.003,0.001]
    betas = [1,0.1,0.01,0.001]
    epsilons = [10,1,0.1,0.01]
    for beta in betas:
        for epsilon in epsilons:
            pmax = -np.Inf
            nmax = -np.Inf
            permax = -np.Inf
            svmmax = -np.Inf
            pname = ''
            nname = ''
            pername = ''
            svmname = ''
            for alpha in alphas:
                names = ["10000p-"+str(alpha)+"-"+str(beta)+"-"+str(epsilon)+".txt",
                        "10000n-"+str(alpha)+"-"+str(beta)+"-"+str(epsilon)+".txt",
                        "10000svm-"+str(alpha)+"-"+str(beta)+".txt",
                        "10000per-"+str(alpha)+"-"+str(beta)+".txt",
                        ]
                ap,norm = getinfo(names[0])
                if ap[-1] > pmax:
                    pmax = ap[-1]
                    pname = names[0]
                ap,norm = getinfo(names[1])
                if ap[-1] > nmax:
                    nmax = ap[-1]
                    nname = names[1]
                ap,norm = getinfo(names[2])
                if ap[-1] > svmmax:
                    svmmax = ap[-1]
                    svmname = names[2]
                ap,norm = getinfo(names[3])
                if ap[-1] > permax:
                    permax = ap[-1]
                    pername = names[3]
                    
            pl.figure()
            names = [pname,nname,svmname,pername]
            for name in names:
                ap,norm = getinfo(name)
                plot(ap,norm)
            pl.legend(['P','N','SVM','PER'])
            name = "combine-beta-"+str(beta)+"-epsilon-"+str(epsilon)+'.png'
            pl.savefig(name,bbox_inches='tight')
            pl.close()
            

if __name__ == '__main__':
    positive()
    negative()
    per()
    svm()
    combine()
    bestAlphaPlot()
    