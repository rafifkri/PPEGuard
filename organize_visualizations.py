"""
Script untuk organize semua output visualization dari Notebook 05
Memindahkan PNG files ke subfolder yang sesuai berdasarkan kategorinya
"""

import os
import shutil
from pathlib import Path

# Define directory structure
DATA_DIR = 'data'
VIZ_DIR = os.path.join(DATA_DIR, 'visualizations')

# Define subdirectories untuk organize visualizations
CATEGORIES = {
    'image_analysis': [
        'advanced_image_analysis.png',
    ],
    'class_analysis': [
        'class_distribution.png',
        'class_balance_detailed.png',
        'class_cooccurrence_matrix.png',
    ],
    'object_analysis': [
        'object_metrics_distribution.png',
    ],
    'quality_analysis': [
        'data_quality_metrics.png',
    ],
    'spatial_analysis': [
        'spatial_distribution.png',
        'spatial_density_heatmap.png',
    ],
    'sample_images': [
        'sample_images_train.png',
        'sample_images_val.png',
        'sample_images_test.png',
    ],
    'summary': [
        'comprehensive_summary_dashboard.png',
    ]
}

def create_directory_structure():
    """Create visualization subdirectories"""
    if not os.path.exists(VIZ_DIR):
        os.makedirs(VIZ_DIR)
        print(f'✓ Created: {VIZ_DIR}')
    
    for category in CATEGORIES.keys():
        category_path = os.path.join(VIZ_DIR, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
            print(f'✓ Created: {category_path}')

def organize_files():
    """Move visualization files to appropriate subdirectories"""
    moved_count = 0
    skipped_count = 0
    
    for category, files in CATEGORIES.items():
        category_path = os.path.join(VIZ_DIR, category)
        
        for filename in files:
            source = os.path.join(DATA_DIR, filename)
            destination = os.path.join(category_path, filename)
            
            if os.path.exists(source):
                if not os.path.exists(destination):
                    shutil.move(source, destination)
                    print(f'✓ Moved: {filename} → {category}/')
                    moved_count += 1
                else:
                    print(f'⚠ Already exists: {filename} in {category}/')
                    skipped_count += 1
            else:
                print(f'✗ Not found: {filename}')
                skipped_count += 1
    
    return moved_count, skipped_count

def print_structure():
    """Print final directory structure"""
    print('\n' + '='*80)
    print('VISUALIZATION DIRECTORY STRUCTURE')
    print('='*80)
    
    for category in sorted(CATEGORIES.keys()):
        category_path = os.path.join(VIZ_DIR, category)
        if os.path.exists(category_path):
            files = os.listdir(category_path)
            print(f'\n📁 {category}/ ({len(files)} files)')
            for file in sorted(files):
                file_size = os.path.getsize(os.path.join(category_path, file)) / 1024  # KB
                print(f'   📄 {file} ({file_size:.1f}KB)')

if __name__ == '__main__':
    print('\n' + '='*80)
    print('ORGANIZING VISUALIZATIONS')
    print('='*80 + '\n')
    
    # 1. Create directory structure
    print('1. Creating directory structure...')
    create_directory_structure()
    
    # 2. Move files
    print('\n2. Moving files to categories...')
    moved, skipped = organize_files()
    
    # 3. Print summary
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)
    print(f'Files moved: {moved}')
    print(f'Files skipped: {skipped}')
    
    # 4. Print final structure
    print_structure()
    
    print('\n✓ Organization complete!')
    print(f'\nAll visualizations organized in: {os.path.abspath(VIZ_DIR)}')
