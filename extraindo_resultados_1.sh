#!/usr/bin/env bash
echo "Começando pós-processamento"
cd /home/daniel/AnacondaProjects/prednet-master_2/py_scripts
source activate tensorflow
while true
do
  # (1) prompt user, and read command line argument
  read -p "Rodar o salvando_resultados_hkl_1.py agora? " answer

  # (2) handle the input we were given
  case $answer in
   [yY]* ) echo "Ok, rodando o script."
           python salvando_resultados_hkl_1.py
           
           break;;

   [nN]* ) break;;

   * )     echo "Digite apenas Y ou N.";;
  esac
done

while true
do
  # (1) prompt user, and read command line argument
  read -p "Rodar o tratando_imgs_jic_2 agora? " answer

  # (2) handle the input we were given
  case $answer in
   [yY]* ) echo "Ok, rodando o script."
           python tratando_imgs_jic_2.py
           
           break;;

   [nN]* ) break;;

   * )     echo "Digite apenas Y ou N.";;
  esac
done
source deactivate tensorflow
source activate cdo
while true
do
  # (1) prompt user, and read command line argument
  read -p "Rodar o comparando_frames_3 agora? " answer

  # (2) handle the input we were given
  case $answer in
   [yY]* ) echo "Ok, rodando o script."
           python comparando_frames_3.py
           
           break;;

   [nN]* ) break;;

   * )     echo "Digite apenas Y ou N.";;
  esac
done
while true
do
  # (1) prompt user, and read command line argument
  read -p "Rodar o comparacao_modelos_A-B.py? " answer

  # (2) handle the input we were given
  case $answer in
   [yY]* ) echo "Ok, rodando o script."
           python comparacao_modelos_A-B.py
           
           break;;

   [nN]* ) break;;

   * )     echo "Digite apenas Y ou N.";;
  esac
done

echo "feito"
