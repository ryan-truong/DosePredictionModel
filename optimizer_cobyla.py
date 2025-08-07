optimizer = Optimizer(r'', 'COBYLA', 'MSE', prescribed_dose, False, r'')
optimizedDoseDistribution = optimizer.minimize(kernel,dcmDose,doseDistribution,patient_aks,np.array(metadata["DT"]))
total_dose = kernel.createSumDose(optimizedDoseDistribution, patient_aks,  False,1)


# More information on the COBYLA optimizer and use can be found here: https://pubmed.ncbi.nlm.nih.gov/36898161/
