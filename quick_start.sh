rm img_result/*

python slam.py --config configs/mono/tum/fr3_office.yaml > output.log 2>&1
# python slam.py --config configs/mono/bonn/fr1_kidnappingbox.yaml > output.log 2>&1