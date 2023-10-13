# SIN5014

copia todas images de dataset_e_codigos para o diretorio dataset_species:

find ../../../../aplicacao/dataset_e_codigos -type f -exec cp -t . {} +

renomear todos os arquivos .JPG para .jpg

for file in \*.JPG; do
mv -- "$file" "${file%}.jpg"
done
