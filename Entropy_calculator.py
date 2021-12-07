from math import log2
def E(p1, n1, p0, n0):
    try:
	    return ((-p1/(p1 + n1) * log2(p1/(p1 + n1)) ) + (-n1/(p1 + n1) * log2(n1/(p1 + n1)) ), 
                (-p0/(p0 + n0) * log2(p0/(p0 + n0)) ) + (-n0/(p0 + n0) * log2(n0/(p0 + n0)) )) 
    except:
        try:
            return (0.0 , 
            (-p0/(p0 + n0) * log2(p0/(p0 + n0)) ) + (-n0/(p0 + n0) * log2(n0/(p0 + n0)) )) 
        except: 
            return ((-p1/(p1 + n1) * log2(p1/(p1 + n1)) ) + (-n1/(p1 + n1) * log2(n1/(p1 + n1)) ), 0.0) 


p1 = int(input("input p1: "))
n1 = int(input("input n1: "))
p0 = int(input("input p0: "))
n0 = int(input("input n0: "))
ES = float(input("input E[S]: "))
print(f"E[S1] = {E(p1, n1, p0, n0)[0]}\nE[S0] = {E(p1, n1, p0, n0)[1]}")
print("IG = ", ES - ((((p1 + n1)/(p1 + n1 + p0 + n0)) * E(p1, n1, p0, n0)[0]) + (((p0 + n0)/(p1 + n1 + p0 + n0)) * E(p1, n1, p0, n0)[1])))

