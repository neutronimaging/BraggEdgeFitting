import math
import matplotlib.pyplot as plt
from numpy import exp, loadtxt, pi, sqrt
import numpy as np
from scipy.signal import argrelextrema
from scipy.special import erfc, erf
from scipy.special import expit

from lmfit import Model

def tau(x):   #This is the Tau-function from Engin-X  
	tau_=(418/(1+155.1*exp(-4.46*x)))/(14342)

	return (tau_)
#print(tau(4.0423))

data1 = loadtxt('Ajuste_test/spectrum_cal_transmission.txt')
data2 = loadtxt('Ajuste_test/mycal_lambda.txt')
x=np.array(data2)
y=np.array(data1)

# First aproximation to find the Lamnda_hkl (peak position)
maximos= (x[argrelextrema (y, np.greater, order=100)])
minimos= (x[argrelextrema (y, np.less, order=100)])
max_pos=argrelextrema (y, np.greater, order=100)
min_pos=argrelextrema (y, np.less, order=100)



iniciales=[]
pos_iniciales=[]
aux_1=[]
aux_2=[]
#
for i,val_i in enumerate(minimos):
	for j,val_j in enumerate(maximos):
		if val_j>val_i:
			iniciales.append((val_i+val_j)/2)
			pos_iniciales.append(np.rint((max_pos[0][j]+min_pos[0][i])/2))
			aux_1.append(max_pos[0][j])
			aux_2.append(min_pos[0][i])
			break
		pass
	pass
pass

val=np.flip(np.array(iniciales))
pos=np.flip(np.array(pos_iniciales))
max_pos=np.flip(np.array(aux_1))
min_pos=np.flip(np.array(aux_2))

#print(val,pos,max_pos,min_pos)

################
#Right fitting
# using 250% (Good shaped peak, a least defined peak need a low percent)

#well, this code was not to be read by others, but let me make an effort for the next two lines

#To do the right fitting i use the 250% of the amount of point between the Bragg Edge center and the maximun
rang_der=np.int32((max_pos[0]-pos[0])*2.5)
#but i also dont use the 10% of the points closer to the maximun
aux_1=np.int32(rang_der*(0.10))


x_der=np.zeros(rang_der-aux_1)
y_der=np.zeros(rang_der-aux_1)

for i in range (rang_der-aux_1):
    x_der[i]=x[max_pos[0]+aux_1+i]
    y_der[i]=y[max_pos[0]+aux_1+i]
    pass

def Ajuste_der (a,x,b):
    return exp(-(a+b*x))

Lineal_model_der= Model (Ajuste_der, independent_vars=['x'],prefix='f1_')
params = Lineal_model_der.make_params(a=-2,b=1)

#Ajuste_model.make_params (a_0=1,b_0=2)

result_der= Lineal_model_der.fit(y_der,params,x=x_der)
#print (result_der.ci_report())
#print (result_der.fit_report())
#print(result_der.best_values.get('f1_a'))

a_0_d=result_der.best_values.get('f1_a')
b_0_d=result_der.best_values.get('f1_b')
print(a_0_d,b_0_d)
#Show the adjust to the right

plt.plot(x_der, y_der,'bo')
plt.plot(x_der, result_der.best_fit,'k--')
plt.show()

#####################
#Left fitting
#This is the same that i do in the right
rang_izq=np.int32((pos[0]-min_pos[0])*2.5) 
aux_1=np.int32(rang_izq*0.4)

x_izq=np.zeros(rang_izq-aux_1)
y_izq=np.zeros(rang_izq-aux_1)

for i in range (rang_izq-aux_1):
    x_izq[i]=x[min_pos[0]-aux_1-i]
    y_izq[i]=y[min_pos[0]-aux_1-i]
    pass

def Ajuste_izq (x,a,b):
    return exp(-(a_0_d+b_0_d*x))*exp(-(a+b*x))


Lineal_model_izq= Model (Ajuste_izq, independent_vars=['x'],prefix='f2_')
params = Lineal_model_izq.make_params(a=-2,b=1)


result_izq= Lineal_model_izq.fit(y_izq,params,x=x_izq)
#print (result_izq.fit_report())

a_0_i=result_izq.best_values.get('f2_a')
b_0_i=result_izq.best_values.get('f2_b')
print(a_0_i,b_0_i)
#Mostrar ajuste a la izq

plt.plot(x_izq, y_izq,'bo')
plt.plot(x_izq, result_izq.best_fit,'k--')
plt.show()


##########################
#last_adjust

rang=np.int32((max_pos[0]-min_pos[0])*3.5)
x_tot=np.zeros(rang)
y_tot=np.zeros(rang)


for i in range (rang):
    x_tot[i]=x[min_pos[0]-rang_izq+i]
    y_tot[i]=y[min_pos[0]-rang_izq+i]
    pass



#Primera Fase

def Ajuste_semifinal (l, sigma,long_hkl):
	return expit(-(a_0_d+b_0_d*l))*expit(-(a_0_i+b_0_i*l))+(expit(-(a_0_d+b_0_d*l))-expit(-(a_0_d+b_0_d*l))*expit(-(a_0_i+b_0_i*l)))*0.5*(erfc(-(l-long_hkl)/(math.sqrt(2)*sigma))-expit(-(l-long_hkl)/(tau(long_hkl))+sigma**2/(2*tau(long_hkl)**2))*erfc(-(l-long_hkl)/(math.sqrt(2)*sigma)+sigma/tau(long_hkl)))

#Here i use a sigma really low, but a good aproximation is use tau(lamnda_hkl)
Semifinal_ajuste=Model(Ajuste_semifinal,independent_vars=['l'], prefix='f3_')
params = Semifinal_ajuste.make_params(sigma=0.002,long_hkl=val[0])

result_semifinal= Semifinal_ajuste.fit(y_tot,params,l=x_tot,nan_policy='propagate')

sigma_0=result_semifinal.best_values.get('f3_sigma')
long_hkl_0=result_semifinal.best_values.get('f3_long_hkl')


#Segunda fase
#to adjust the sigma
def Ajuste_final (l,a_0, b_0, a, b, sigma,long_hkl):
	return expit(-(a_0+b_0*l))*expit(-(a+b*l))+(expit(-(a_0+b_0*l))-expit(-(a_0+b_0*l))*expit(-(a+b*l)))*0.5*(erfc(-(l-long_hkl)/(math.sqrt(2)*sigma))-expit(-(l-long_hkl)/(tau(long_hkl))+sigma**2/(2*tau(long_hkl)**2))*erfc(-(l-long_hkl)/(math.sqrt(2)*sigma)+sigma/tau(long_hkl)))


Final_model=Model(Ajuste_final,independent_vars=['l'], prefix='f4_')	
params = Final_model.make_params(a_0=a_0_d,b_0=b_0_d,a=a_0_i,b=b_0_i,sigma=sigma_0,long_hkl=long_hkl_0)

last_result= Final_model.fit(y_tot,params,l=x_tot,nan_policy='propagate')





h=(last_result.best_values.get('f4_a_0')+last_result.best_values.get('f4_b_0')*last_result.best_values.get('f4_long_hkl'))-(last_result.best_values.get('f4_a')+last_result.best_values.get('f4_b')*last_result.best_values.get('f4_long_hkl'))
print("Chi:"+ str(last_result.redchi) + "\n Center:"+ str(last_result.best_values.get('f4_long_hkl'))+"\n Bragg border height: "+str(-h))

#Mostrar el mejor ajuste 
plt.plot(x_tot, y_tot,'bo')
#plt.plot(x_tot, last_result.init_fit, 'k--')
plt.plot(x_tot, last_result.best_fit,color='red', linewidth=3)
#plt.plot(x_tot, result_semifinal.best_fit,color='green', linewidth=3) #1st aprox
plt.show()
