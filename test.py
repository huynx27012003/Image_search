from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
higher_education_students_performance_evaluation = fetch_ucirepo(id=856) 
  
# data (as pandas dataframes) 
X = higher_education_students_performance_evaluation.data.features 
y = higher_education_students_performance_evaluation.data.targets 
  
# metadata 
print(higher_education_students_performance_evaluation.metadata) 
  
# variable information 
print(higher_education_students_performance_evaluation.variables) 
