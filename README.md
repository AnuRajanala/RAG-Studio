RAG Studio Streamlit App Installation and Running
1. Checkout the git repo
2. Navigate to Code
3. Run
     1. pip install -e . (. is part of the command)
     2. streamlit run FrontEnd\Main.py

Note:
1. If any files have new dependencies - please update requirements.in
2. If file has pure Back end logic - keep in BackEnd folder
    else
    Keep in Front End Folder

Necessary steps to connect to OCI-
1. Login cloud.oracle.com
2. Navigate to your profile and select tokens and keys tab.
3. Click on add API key and select 'Generate API Key pair'. Download the public key and private key.
4. A new fingerprint gets added. Select 'view configuration file' and paste the contents to a file with name 'config' in the following location -
	\code (RAG Studio location)
	
Pasting note which is shown after generating API key for reference
Note: This configuration file snippet includes the basic authentication information you'll need to use the SDK, CLI, or other OCI developer tool. Paste the contents of the text box into your ~/.oci/config file and update the key_file parameter with the file path to your private key.
