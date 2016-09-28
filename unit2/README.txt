Bidirectional RNN based sorting model

1. accuracy.py - script to find accuracy percentage between therotical and RNN values
    python accuracy.py SizeOfDataSet TrialNo
    Eg : python accuracy.py 10000 2 
    
2. create_data.py - script to generate input Datasets
    python create_data.py SizeOfDataSet
    
3. inputs - Directory containing input datasets for training and test  

4. models - Directory containing pickle binaries of built models

5. results - Directory containing therotical results and RNN results

6. sort.py - Build model and dump to pickle file in models directory
    python sort.py SizeOfDataSet
    
7. test_file1.py - Run test script using approach 1 [Mentioned in the Doc on FaceBook Page]
    python test_file1.py SizeOfDataSet
    
8. test_file2.py - Run test script using approcah 2 [Mentioned in the Doc]
    python test_file2.py SizeOfDataSet                         
