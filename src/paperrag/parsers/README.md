# PaperRAG Parsers

This repository automates the setup and operation of multiple document parsing services using an entrypoint script (`entrypoint.sh`). The services include:

- **llmsherpa**: Document parsers also used in Azure  
- **pdffigures2**: A Scala-based figure extraction service  

When you run:
```bash
chmod +x entrypoint.sh
./entrypoint
cd src/parsers