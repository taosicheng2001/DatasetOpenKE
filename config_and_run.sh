git clone https://github.com/thunlp/OpenKE.git
cd OpenKE && cd openke && bash make.sh && cd ../../
git clone https://github.com/tusenchun/DatasetOpenKE.git
cp -r DatasetOpenKE -r OpenKE/
cd OpenKE && cp DatasetOpenKE/transh_dataset.py ./ && cd  ..
cd OpenKE && mkdir checkopint && ls && cd ..
cd OpenKE && python3 transh_dataset.py 