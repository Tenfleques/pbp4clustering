#!/usr/bin/env python3
"""
Dataset Information Database for PBP Analysis

This module contains comprehensive information about all datasets used in the
Pseudo-Boolean Polynomial dimensionality reduction evaluation, including:
- Dataset descriptions and purposes
- Feature/column descriptions
- Target variable explanations
- PBP transformation rationale
- Data types and domains
"""

DATASET_INFO = {
    # Standard Datasets
    'adult': {
        'name': 'Adult Census Income',
        'description': 'Predict whether income exceeds $50K/yr based on census data. This dataset contains demographic and employment information from the 1994 Census database.',
        'domain': 'social_real',
        'source': 'UCI Machine Learning Repository',
        'features': {
            'age': 'Age of the individual',
            'workclass': 'Type of employer (private, government, etc.)',
            'fnlwgt': 'Final weight (sampling weight)',
            'education': 'Level of education',
            'education_num': 'Numeric education level',
            'marital_status': 'Marital status',
            'occupation': 'Type of occupation',
            'relationship': 'Relationship status',
            'race': 'Race of the individual',
            'sex': 'Gender',
            'capital_gain': 'Capital gains',
            'capital_loss': 'Capital losses',
            'hours_per_week': 'Hours worked per week',
            'native_country': 'Country of origin'
        },
        'target': {
            'name': 'income',
            'description': 'Binary classification: income > $50K (1) or <= $50K (0)',
            'classes': ['<=50K', '>50K']
        },
        'pbp_transformation': {
            'matrix_shape': '(4, 5)',
            'rationale': 'Reshaped into 4x5 matrices where rows represent demographic categories (Personal, Employment, Financial, Geographic) and columns represent specific measurements within each category.',
            'feature_names': ['Personal', 'Employment', 'Financial', 'Geographic'],
            'measurement_names': ['Demographic_1', 'Demographic_2', 'Demographic_3', 'Demographic_4', 'Demographic_5']
        },
        'sample_count': 1109
    },
    
    'breast_cancer_sklearn': {
        'name': 'Wisconsin Breast Cancer',
        'description': 'Diagnosis of breast cancer as malignant or benign based on cell nucleus characteristics. Features are computed from a digitized image of a fine needle aspirate of a breast mass.',
        'domain': 'medical_real',
        'source': 'scikit-learn built-in dataset',
        'features': {
            'mean_radius': 'Mean radius of cell nucleus',
            'mean_texture': 'Mean texture of cell nucleus',
            'mean_perimeter': 'Mean perimeter of cell nucleus',
            'mean_area': 'Mean area of cell nucleus',
            'mean_smoothness': 'Mean smoothness of cell nucleus',
            'mean_compactness': 'Mean compactness of cell nucleus',
            'mean_concavity': 'Mean concavity of cell nucleus',
            'mean_concave_points': 'Mean concave points of cell nucleus',
            'mean_symmetry': 'Mean symmetry of cell nucleus',
            'mean_fractal_dimension': 'Mean fractal dimension of cell nucleus',
            'se_radius': 'Standard error of radius',
            'se_texture': 'Standard error of texture',
            'se_perimeter': 'Standard error of perimeter',
            'se_area': 'Standard error of area',
            'se_smoothness': 'Standard error of smoothness',
            'se_compactness': 'Standard error of compactness',
            'se_concavity': 'Standard error of concavity',
            'se_concave_points': 'Standard error of concave points',
            'se_symmetry': 'Standard error of symmetry',
            'se_fractal_dimension': 'Standard error of fractal dimension',
            'worst_radius': 'Worst radius of cell nucleus',
            'worst_texture': 'Worst texture of cell nucleus',
            'worst_perimeter': 'Worst perimeter of cell nucleus',
            'worst_area': 'Worst area of cell nucleus',
            'worst_smoothness': 'Worst smoothness of cell nucleus',
            'worst_compactness': 'Worst compactness of cell nucleus',
            'worst_concavity': 'Worst concavity of cell nucleus',
            'worst_concave_points': 'Worst concave points of cell nucleus',
            'worst_symmetry': 'Worst symmetry of cell nucleus',
            'worst_fractal_dimension': 'Worst fractal dimension of cell nucleus'
        },
        'target': {
            'name': 'target',
            'description': 'Binary classification: malignant (1) or benign (0)',
            'classes': ['benign', 'malignant']
        },
        'pbp_transformation': {
            'matrix_shape': '(5, 6)',
            'rationale': 'Reshaped into 5x6 matrices where rows represent different statistical measures (Mean, SE, Worst) and columns represent different cell nucleus characteristics.',
            'feature_names': ['Mean', 'SE', 'Worst'],
            'measurement_names': ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness']
        },
        'sample_count': 569
    },
    
    'churn': {
        'name': 'Customer Churn',
        'description': 'Predict customer churn based on customer behavior and demographics. Contains information about customers who left within the last month.',
        'domain': 'business_real',
        'source': 'Synthetic business dataset',
        'features': {
            'credit_score': 'Customer credit score',
            'geography': 'Customer location',
            'gender': 'Customer gender',
            'age': 'Customer age',
            'tenure': 'Number of years with the bank',
            'balance': 'Account balance',
            'num_of_products': 'Number of bank products used',
            'has_cr_card': 'Has credit card (1/0)',
            'is_active_member': 'Is active member (1/0)',
            'estimated_salary': 'Estimated salary',
            'exited': 'Churned (1) or stayed (0)'
        },
        'target': {
            'name': 'exited',
            'description': 'Binary classification: customer churned (1) or stayed (0)',
            'classes': ['stayed', 'churned']
        },
        'pbp_transformation': {
            'matrix_shape': '(4, 5)',
            'rationale': 'Reshaped into 4x5 matrices where rows represent customer categories (Demographics, Financial, Behavioral, Engagement) and columns represent specific measurements.',
            'feature_names': ['Demographics', 'Financial', 'Behavioral', 'Engagement'],
            'measurement_names': ['Customer_1', 'Customer_2', 'Customer_3', 'Customer_4', 'Customer_5']
        },
        'sample_count': 5000
    },
    
    'credit_approval': {
        'name': 'Credit Card Approval',
        'description': 'Predict credit card approval based on customer information and financial history.',
        'domain': 'financial_real',
        'source': 'Financial institution dataset',
        'features': {
            'feature_1': 'Customer demographic feature',
            'feature_2': 'Financial history feature',
            'feature_3': 'Credit behavior feature',
            'feature_4': 'Employment feature',
            'feature_5': 'Income feature',
            'feature_6': 'Approval status'
        },
        'target': {
            'name': 'approval',
            'description': 'Binary classification: approved (1) or denied (0)',
            'classes': ['denied', 'approved']
        },
        'pbp_transformation': {
            'matrix_shape': '(2, 3)',
            'rationale': 'Reshaped into 2x3 matrices where rows represent customer categories (Personal, Financial) and columns represent specific measurements.',
            'feature_names': ['Personal', 'Financial'],
            'measurement_names': ['Credit_1', 'Credit_2', 'Credit_3']
        },
        'sample_count': 1000
    },
    
    'diabetes': {
        'name': 'Pima Indians Diabetes',
        'description': 'Predict diabetes diagnosis based on diagnostic measurements. Contains medical diagnostic data from Pima Indian heritage.',
        'domain': 'medical_real',
        'source': 'UCI Machine Learning Repository',
        'features': {
            'pregnancies': 'Number of times pregnant',
            'glucose': 'Plasma glucose concentration',
            'blood_pressure': 'Diastolic blood pressure',
            'skin_thickness': 'Triceps skin fold thickness',
            'insulin': '2-Hour serum insulin',
            'bmi': 'Body mass index',
            'diabetes_pedigree': 'Diabetes pedigree function',
            'age': 'Age in years'
        },
        'target': {
            'name': 'outcome',
            'description': 'Binary classification: diabetic (1) or non-diabetic (0)',
            'classes': ['non-diabetic', 'diabetic']
        },
        'pbp_transformation': {
            'matrix_shape': '(2, 4)',
            'rationale': 'Reshaped into 2x4 matrices where rows represent metabolic categories (Metabolic, Reproductive) and columns represent specific diagnostic measurements.',
            'feature_names': ['Metabolic', 'Reproductive'],
            'measurement_names': ['Diagnostic_1', 'Diagnostic_2', 'Diagnostic_3', 'Diagnostic_4']
        },
        'sample_count': 768
    },
    
    'digits_sklearn': {
        'name': 'Handwritten Digits',
        'description': 'Recognize handwritten digits (0-9) from 8x8 pixel images. Each image is a grayscale digit from 0 to 9.',
        'domain': 'image_real',
        'source': 'scikit-learn built-in dataset',
        'features': {
            'pixel_0_0': 'Pixel value at position (0,0)',
            'pixel_0_1': 'Pixel value at position (0,1)',
            # ... 64 total pixel features
            'pixel_7_7': 'Pixel value at position (7,7)'
        },
        'target': {
            'name': 'target',
            'description': 'Multi-class classification: digit 0-9',
            'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        },
        'pbp_transformation': {
            'matrix_shape': '(5, 6)',
            'rationale': 'Reshaped into 5x6 matrices where rows represent image regions (Top-Left, Top-Right, Center, Bottom-Left, Bottom-Right) and columns represent pixel measurements.',
            'feature_names': ['Top-Left', 'Top-Right', 'Center', 'Bottom-Left', 'Bottom-Right'],
            'measurement_names': ['Pixel_1', 'Pixel_2', 'Pixel_3', 'Pixel_4', 'Pixel_5', 'Pixel_6']
        },
        'sample_count': 1797
    },
    
    'pc4': {
        'name': 'PC4 Software Defects',
        'description': 'Predict software defects in NASA PC4 project based on code metrics and complexity measures.',
        'domain': 'software_real',
        'source': 'NASA Metrics Data Program',
        'features': {
            'loc': 'Lines of code',
            'v': 'McCabe cyclomatic complexity',
            'ev': 'Essential complexity',
            'iv': 'Design complexity',
            'n': 'Halstead program length',
            'v': 'Halstead volume',
            'l': 'Halstead program level',
            'd': 'Halstead difficulty',
            'i': 'Halstead intelligence',
            'e': 'Halstead effort',
            'b': 'Halstead time estimator',
            't': 'Halstead time',
            'lOCode': 'Lines of code',
            'lOComment': 'Lines of comment',
            'lOBlank': 'Lines of blank',
            'lOCodeAndComment': 'Lines of code and comment',
            'uniq_Op': 'Unique operators',
            'uniq_Opnd': 'Unique operands',
            'total_Op': 'Total operators',
            'total_Opnd': 'Total operands',
            'branchCount': 'Branch count'
        },
        'target': {
            'name': 'defects',
            'description': 'Binary classification: defective (1) or non-defective (0)',
            'classes': ['non-defective', 'defective']
        },
        'pbp_transformation': {
            'matrix_shape': '(5, 6)',
            'rationale': 'Reshaped into 5x6 matrices where rows represent code quality categories (Complexity, Volume, Length, Effort, Structure) and columns represent specific metrics.',
            'feature_names': ['Complexity', 'Volume', 'Length', 'Effort', 'Structure'],
            'measurement_names': ['Metric_1', 'Metric_2', 'Metric_3', 'Metric_4', 'Metric_5', 'Metric_6']
        },
        'sample_count': 1458
    },
    
    'wine_quality_red': {
        'name': 'Red Wine Quality',
        'description': 'Predict wine quality based on physicochemical properties. Contains sensory data for red wine samples.',
        'domain': 'chemical_real',
        'source': 'UCI Machine Learning Repository',
        'features': {
            'fixed_acidity': 'Fixed acidity (tartaric acid)',
            'volatile_acidity': 'Volatile acidity (acetic acid)',
            'citric_acid': 'Citric acid content',
            'residual_sugar': 'Residual sugar content',
            'chlorides': 'Chloride content',
            'free_sulfur_dioxide': 'Free sulfur dioxide',
            'total_sulfur_dioxide': 'Total sulfur dioxide',
            'density': 'Density of wine',
            'ph': 'pH value',
            'sulphates': 'Sulphate content',
            'alcohol': 'Alcohol content'
        },
        'target': {
            'name': 'quality',
            'description': 'Multi-class classification: wine quality 3-8 (3=low, 8=high)',
            'classes': ['3', '4', '5', '6', '7', '8']
        },
        'pbp_transformation': {
            'matrix_shape': '(2, 5)',
            'rationale': 'Reshaped into 2x5 matrices where rows represent chemical categories (Acids, Alcohols) and columns represent specific chemical measurements.',
            'feature_names': ['Acids', 'Alcohols'],
            'measurement_names': ['Chemical_1', 'Chemical_2', 'Chemical_3', 'Chemical_4', 'Chemical_5']
        },
        'sample_count': 1599
    },
    
    'wine_sklearn': {
        'name': 'Wine Recognition',
        'description': 'Classify wines into three cultivars based on chemical analysis of the wine.',
        'domain': 'chemical_real',
        'source': 'scikit-learn built-in dataset',
        'features': {
            'alcohol': 'Alcohol content',
            'malic_acid': 'Malic acid content',
            'ash': 'Ash content',
            'alcalinity_of_ash': 'Alkalinity of ash',
            'magnesium': 'Magnesium content',
            'total_phenols': 'Total phenols',
            'flavanoids': 'Flavanoids content',
            'nonflavanoid_phenols': 'Nonflavanoid phenols',
            'proanthocyanins': 'Proanthocyanins',
            'color_intensity': 'Color intensity',
            'hue': 'Hue',
            'od280_od315_of_diluted_wines': 'OD280/OD315 of diluted wines',
            'proline': 'Proline content'
        },
        'target': {
            'name': 'target',
            'description': 'Multi-class classification: wine cultivar 0, 1, or 2',
            'classes': ['cultivar_0', 'cultivar_1', 'cultivar_2']
        },
        'pbp_transformation': {
            'matrix_shape': '(3, 4)',
            'rationale': 'Reshaped into 3x4 matrices where rows represent chemical categories (Acids, Alcohols, Phenols) and columns represent specific measurements.',
            'feature_names': ['Acids', 'Alcohols', 'Phenols'],
            'measurement_names': ['Chemical_1', 'Chemical_2', 'Chemical_3', 'Chemical_4']
        },
        'sample_count': 178
    },
    
    # Advanced Real Datasets
    'gdsc_expression': {
        'name': 'GDSC Gene Expression',
        'description': 'Gene expression data from cancer cell lines in the Genomics of Drug Sensitivity in Cancer (GDSC) project. Contains expression levels of genes across different cancer cell lines.',
        'domain': 'gene_expression_real',
        'source': 'GDSC Cancer Genomics (E-MTAB-3610)',
        'features': {
            'gene_expression': 'Expression levels of various genes across cancer cell lines',
            'cell_line_id': 'Unique identifier for each cancer cell line',
            'gene_id': 'Gene identifiers',
            'expression_value': 'Normalized expression values'
        },
        'target': {
            'name': 'cell_line_type',
            'description': 'Multi-class classification: cancer cell line types (807 unique types)',
            'classes': '807 unique cancer cell line types'
        },
        'pbp_transformation': {
            'matrix_shape': '(5, 8)',
            'rationale': 'Reshaped into 5x8 matrices where rows represent gene expression categories and columns represent specific gene measurements.',
            'feature_names': ['Gene_Category_1', 'Gene_Category_2', 'Gene_Category_3', 'Gene_Category_4', 'Gene_Category_5'],
            'measurement_names': ['Gene_1', 'Gene_2', 'Gene_3', 'Gene_4', 'Gene_5', 'Gene_6', 'Gene_7', 'Gene_8']
        },
        'sample_count': 811
    },
    
    'metabolights_mtbls1': {
        'name': 'MetaboLights MTBLS1',
        'description': 'Metabolomics data comparing diabetes patients vs. controls. Contains NMR spectroscopy metabolite concentration data.',
        'domain': 'metabolomics_real',
        'source': 'MetaboLights MTBLS1',
        'features': {
            'metabolite_concentration': 'Concentration of various metabolites',
            'sample_id': 'Sample identifiers',
            'metabolite_id': 'Metabolite identifiers',
            'concentration_value': 'NMR spectroscopy concentration values'
        },
        'target': {
            'name': 'disease_status',
            'description': 'Binary classification: diabetes (1) or control (0)',
            'classes': ['control', 'diabetes']
        },
        'pbp_transformation': {
            'matrix_shape': '(4, 4)',
            'rationale': 'Reshaped into 4x4 matrices where rows represent metabolite categories and columns represent specific metabolite measurements.',
            'feature_names': ['Metabolite_Category_1', 'Metabolite_Category_2', 'Metabolite_Category_3', 'Metabolite_Category_4'],
            'measurement_names': ['Metabolite_1', 'Metabolite_2', 'Metabolite_3', 'Metabolite_4']
        },
        'sample_count': 132
    },
    
    'nci_chemical': {
        'name': 'NCI Chemical Database',
        'description': 'Molecular descriptors and properties of chemical compounds from the National Cancer Institute database. Contains SMILES strings converted to molecular fingerprints and various molecular properties.',
        'domain': 'chemical_real',
        'source': 'NCI Chemical Database',
        'features': {
            'molecular_fingerprints': 'Morgan fingerprints derived from SMILES strings',
            'molecular_weight': 'Molecular weight of compounds',
            'cas_number': 'CAS registry numbers',
            'nsc_number': 'NSC (National Service Center) numbers'
        },
        'target': {
            'name': 'molecular_weight_class',
            'description': 'Binary classification: high molecular weight (1) or low molecular weight (0) based on median MW',
            'classes': ['low_mw', 'high_mw']
        },
        'pbp_transformation': {
            'matrix_shape': '(4, 8)',
            'rationale': 'Reshaped into 4x8 matrices where rows represent molecular descriptor categories and columns represent specific molecular measurements.',
            'feature_names': ['Molecular_Descriptor_1', 'Molecular_Descriptor_2', 'Molecular_Descriptor_3', 'Molecular_Descriptor_4'],
            'measurement_names': ['Descriptor_1', 'Descriptor_2', 'Descriptor_3', 'Descriptor_4', 'Descriptor_5', 'Descriptor_6', 'Descriptor_7', 'Descriptor_8']
        },
        'sample_count': 4859
    },
    
    'aids_screen': {
        'name': 'AIDS Antiviral Screen',
        'description': 'Antiviral activity screening data from the National Cancer Institute. Contains EC50 and IC50 values for compounds tested against HIV-1, along with experimental statistics.',
        'domain': 'antiviral_real',
        'source': 'NCI AIDS Antiviral Screen Database',
        'features': {
            'log10ec50': 'Log10 of 50% effective concentration (EC50)',
            'log10ic50': 'Log10 of 50% inhibitory concentration (IC50)',
            'numexp_ec50': 'Number of experiments for EC50 measurement',
            'numexp_ic50': 'Number of experiments for IC50 measurement',
            'stddev_ec50': 'Standard deviation of EC50 measurements',
            'stddev_ic50': 'Standard deviation of IC50 measurements',
            'screening_result': 'Antiviral activity classification'
        },
        'target': {
            'name': 'screening_result',
            'description': 'Multi-class classification: CI (inactive), CM (moderately active), CA (active)',
            'classes': ['CI', 'CM', 'CA']
        },
        'pbp_transformation': {
            'matrix_shape': '(2, 4)',
            'rationale': 'Reshaped into 2x4 matrices where rows represent measurement categories (EC50_Data, IC50_Data) and columns represent specific antiviral measurements.',
            'feature_names': ['EC50_Data', 'IC50_Data'],
            'measurement_names': ['Antiviral_1', 'Antiviral_2', 'Antiviral_3', 'Antiviral_4']
        },
        'sample_count': 49556
    }
}

def get_dataset_info(dataset_name):
    """Get comprehensive information about a dataset."""
    return DATASET_INFO.get(dataset_name, None)

def get_all_dataset_names():
    """Get list of all available dataset names."""
    return list(DATASET_INFO.keys())

def get_datasets_by_domain(domain):
    """Get all datasets in a specific domain."""
    return {name: info for name, info in DATASET_INFO.items() 
            if info['domain'] == domain}

def get_domain_summary():
    """Get summary of datasets by domain."""
    domains = {}
    for name, info in DATASET_INFO.items():
        domain = info['domain']
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(name)
    return domains

if __name__ == "__main__":
    """Print dataset information summary."""
    print("Dataset Information Database")
    print("=" * 50)
    
    domains = get_domain_summary()
    for domain, datasets in domains.items():
        print(f"\n{domain.upper()} DATASETS ({len(datasets)}):")
        for dataset in datasets:
            info = DATASET_INFO[dataset]
            print(f"  - {dataset}: {info['name']} ({info['sample_count']} samples)")
    
    print(f"\nTotal datasets: {len(DATASET_INFO)}") 