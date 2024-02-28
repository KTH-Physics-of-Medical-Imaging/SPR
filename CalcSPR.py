#Base script for SPR calculations (needs CatSim installed in venv)
import numpy as np
from catsim.pyfiles.ReadMaterialFile import ReadMaterialFile

#Global variables
m_e = 9.1093837015e-31 #Rest mass electron, from Näsmark
eV_to_joules = 1.602e-19 #Conversion factor for I in eV
N_A = 6.02214076e+23 #Avogadro's number, Britannica
c=299792458 #Speed of light
beta = 0.461376419  #Given 100 MeV protons (Näsmark)

n_Mayneord = 3.21

e_density_water = 3.3431243734025044e+23 #Calculated with below functions for ncat_water =ICRU 46 equivalent
i_water = 78.26905277899314 #Calculated with below functions for ncat_water

mt_path = 'C:/Users/Karin/CatSim/catsim/material/'

atomic_weight = {1: 1.008, 5: 10.81, 6: 12.011, 7: 14.007, 8: 15.999, 11: 22.990, 12: 24.305, 14: 28.085, 15: 30.974, 16: 32.066, 17: 35.453,
                 19: 39.098, 20: 40.078, 26: 55.845, 53: 126.905, 56: 137.327}  # from NIST

ionization_energy = {1: 22.07, 5: 85.88, 6: 79.91, 7: 77.91, 8: 107.44, 11: 168.37, 12: 176.28, 14: 150.47, 15: 199.39, 16: 203.4, 17: 175.13,
                     19: 214.7, 20: 258.11, 26: 323.18, 53: 554.83, 56: 554.83}  # Bär and ICRU rep 37

def mean_ionization_E(materialName):
    #Used by value_SPR, calculate the mean ionization energy of the material (mix of elements)
    numberOfElements, density, atomicNumbers, massFractions = ReadMaterialFile(mt_path + materialName)
    I_nom_sum = 0
    I_denom_sum = 0
    for i in range(numberOfElements):
        I_nom_sum += (massFractions[i] * atomicNumbers[i] * np.log(ionization_energy[atomicNumbers[i]]))/atomic_weight[atomicNumbers[i]]
        I_denom_sum += (massFractions[i] * atomicNumbers[i])/atomic_weight[atomicNumbers[i]]

    return np.exp(I_nom_sum/I_denom_sum)

def EAN(materialName):
    numberOfElements, density, atomicNumbers, massFractions = ReadMaterialFile(mt_path + materialName)
    EAN_nom_sum = 0
    EAN_denom_sum = 0
    for i in range(numberOfElements):
        EAN_nom_sum += (massFractions[i] * atomicNumbers[i] * atomicNumbers[i]**n_Mayneord)/atomic_weight[atomicNumbers[i]]
        EAN_denom_sum += (massFractions[i] * atomicNumbers[i])/atomic_weight[atomicNumbers[i]]

    return (EAN_nom_sum/EAN_denom_sum)**(1/n_Mayneord)

def Electron_Density(materialName):
    numberOfElements, density, atomicNumbers, massFractions = ReadMaterialFile(mt_path + materialName)
    e_density = 0
    for i in range(numberOfElements):
        e_density += (massFractions[i] * atomicNumbers[i])/atomic_weight[atomicNumbers[i]]

    return (density*N_A*e_density)
def rel_Electron_Density(materialName):
    numberOfElements, density, atomicNumbers, massFractions = ReadMaterialFile(mt_path + materialName)
    e_density = 0
    for i in range(numberOfElements):
        e_density += (massFractions[i] * atomicNumbers[i])/atomic_weight[atomicNumbers[i]]

    return (density*N_A*e_density)/e_density_water

def value_SPR(materialName):
    SPR = rel_Electron_Density(materialName) \
          * (np.log((2 * m_e * c ** 2 * beta ** 2) / (
                (1 - beta ** 2) * mean_ionization_E(materialName)*eV_to_joules)) - beta ** 2) / \
          (np.log((2 * m_e * c ** 2 * beta ** 2) / ((1 - beta ** 2) * i_water * eV_to_joules)) - beta ** 2)
    return SPR

if __name__ == "__main__":
    material = 'karin_white_matter'
    numberOfElements, density, atomicNumbers, massFractions = ReadMaterialFile(mt_path + material)
    print('density:', density)
    print('EAN:', EAN(material))
    print('ED:', Electron_Density(material))
    print('I:', mean_ionization_E(material))
    print('RED:', rel_Electron_Density(material))
    print('SPR:', value_SPR(material))