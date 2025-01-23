# document_loader.py
import asyncio
import logging
import os
from typing import Tuple, List, Dict, Any
from paperrag.document.pdf_parser import parse_files
from paperrag.document.document import Document
import traceback
from paperrag.document.figure_parser import FigureExtractor

logger = logging.getLogger(__name__)

# TODO: Need to update the DocumentLoader class to  process tables from llmsherpa; as those are also saved seperate documents

class DocumentLoader:
    def __init__(self, pdf_path: str = None, output_dir: str = None, async_mode: bool = True):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.async_mode = async_mode
        self.figure_extractor = FigureExtractor(pdf_path, output_dir)
        logging.debug(f"Initialized DocumentLoader with pdf_path={pdf_path}, output_dir={output_dir}, async_mode={async_mode}")

    def convert_to_document_objects(self, docs: List[Dict[str, Any]], figures: List[Dict[str, Any]]) -> List[Document]:
        logging.debug("Converting to Document objects.")
        documents = []
        
        # Create doc index to source mapping
        source_mapping = {i: fig.get("document", f"doc_{i}") for i, fig in enumerate(figures)}
        
        # Process documents
        for i, doc_item in enumerate(docs):
            source = source_mapping.get(i, f"doc_{i}")
            logging.debug(f"Processing doc index {i}, source={source}")
            
            # Initialize counters for this document
            type_counters = {
                'chunk': 0,
                'text': 0, 
                'Table': 0,
                'Figure': 0
            }

            try:
                chunks = getattr(doc_item, 'chunks', lambda: [])()
                
                # Process chunks
                for chunk in chunks:
                    try:
                        content = chunk.to_text(include_children=True) if hasattr(chunk, 'to_text') else str(chunk)
                        page_idx = getattr(chunk, 'page_idx', 0)
                        type_counters['chunk'] += 1
                        
                        metadata = {
                            'page_idx': page_idx,
                            'section_title': getattr(chunk, 'parent_text', lambda: None)(),
                            'type': 'chunk',
                            'source': source,
                            'id': f"{source}_p{page_idx}_chunk_{type_counters['chunk']}"
                        }
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)
                    except Exception as e:
                        logging.error(f"Error processing chunk: {e}")
                        continue

                # Process tables
                for table in getattr(doc_item, 'tables', lambda: [])():
                    try:
                        content = table.to_text(include_children=True) if hasattr(table, 'to_text') else str(table)
                        page_idx = getattr(table, 'page_idx', 0)
                        type_counters['Table'] += 1
                        
                        metadata = {
                            'page_idx': page_idx,
                            'section_title': getattr(table, 'parent_text', lambda: None)(),
                            'type': 'Table',
                            'source': source,
                            'id': f"{source}_p{page_idx}_table_{type_counters['Table']}"
                        }
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)
                    except Exception as e:
                        logging.error(f"Error processing table: {e}")
                        continue
            except Exception as e:
                logging.error(f"Error processing document {i}: {e}")
                continue

        # Process figures // TO UPDATE
        for i, figure in enumerate(figures):
            source = figure.get("document", f"doc_{i}")
            type_counters = {'Figure': 0}  # Reset figure counter for each document
            
            try:
                logging.debug(f"Processing figure index {i}, source={source}")
                for figure_path in figure.get("figures", []):
                    try:
                        filename = os.path.basename(figure_path)
                        matching_metadata = next(
                            (item for item in figure.get("figures_with_metadata", []) 
                            if item.get("figure") == filename),
                            None
                        )
                        if matching_metadata:
                            type_counters['Figure'] += 1
                            fig_type = matching_metadata.get("figType", "Figure")
                            page_idx = matching_metadata.get("metadata", {}).get("page", 0)
                            
                            metadata = {
                                'figure': matching_metadata["figure"],
                                'source': source,
                                'id': f"{source}_p{page_idx}_{fig_type.lower()}_{type_counters['Figure']}",
                                **{k: v for k, v in matching_metadata.get("metadata", {}).items() 
                                if k != "renderDpi"}
                            }
                            figure_document = Document(
                                page_content=figure_path,
                                metadata=metadata,
                                type=fig_type
                            )
                            documents.append(figure_document)
                    except Exception as e:
                        logging.error(f"Error processing figure {figure_path}: {e}")
                        continue
            except Exception as e:
                logging.error(f"Error processing figure block {i}: {e}")
                continue

        logging.debug(f"Converted {len(documents)} total documents")
        return documents

    async def load_async(self) -> Tuple[List[Document], Dict[str, Any]]:
        logging.info("Starting asynchronous loading process.")
        try:
            # Step 1: Process PDFs and extract figures concurrently
            logging.debug("Starting PDF processing and figure extraction...")
            docs_task = parse_files(self.pdf_path)
            figures_task = asyncio.to_thread(self.figure_extractor.extract)
            
            docs, figure_response = await asyncio.gather(docs_task, figures_task)
            logging.debug(f"PDF processing complete: {len(docs) if docs else 0} documents")
            logging.debug(f"Figure extraction complete: {figure_response}")
            logging.debug(f"Figure extraction complete: {len(figure_response) if figure_response else 0} figures")
            
            # Step 2: Validate inputs
            if not docs:
                logging.warning("No documents were processed")
                return [], {}

            if not figure_response:
                logging.warning("No figures were processed")
                figure_response = []

            # Step 3: Calculate stats
            try:
                stats = self._calculate_stats(docs, figure_response)
                logging.debug(f"Stats calculated successfully: {stats}")
            except Exception as stats_error:
                logging.error(f"Error calculating stats: {stats_error}")
                stats = {}

            # Step 4: Convert documents
            logging.debug("About to convert documents...")
            try:
                documents = self.convert_to_document_objects(docs, figure_response)
                logging.debug(f"Documents converted successfully: {len(documents)}")
            except Exception as conv_error:
                logging.error(f"Error converting documents: {conv_error}")
                documents = []

            logging.info(f"Asynchronous loading completed with {len(documents)} documents")
            return documents, stats

        except Exception as e:
            logging.error(f"Error during asynchronous loading: {e}")
            logging.debug(traceback.format_exc())
            return [], {}
                
    def load_sync(self) -> Tuple[List[Document], Dict[str, Any]]:
        logging.info("Starting synchronous loading process.")
        try:
            # Run the coroutine in a synchronous context
            docs = asyncio.run(parse_files(self.pdf_path))
            logging.debug(f"Processed PDFs: {docs}")
            
            figure_response = self.figure_extractor.extract()
            logging.debug(f"Extracted figures: {figure_response}")

            if not docs:
                logging.warning("No documents were processed")
                return [], {}

            if not figure_response:
                logging.warning("No figures were processed")
                figure_response = []

            logging.debug("About to calculate stats...")
            stats = self._calculate_stats(docs, figure_response)
            logging.debug(f"Stats calculated: {stats}")
            
            logging.debug("About to convert documents...")
            documents = self.convert_to_document_objects(docs, figure_response)
            logging.debug(f"Documents converted: {len(documents)}")
            
            logging.debug("About to print stats...")
            print(f"Stats: {stats}")
            
            logging.info(f"Synchronous loading completed successfully with {len(documents)} documents")
            return documents, stats

        except Exception as e:
            logging.error(f"Error during synchronous loading: {e}")
            logging.error(f"Exception traceback: {traceback.format_exc()}")
            return [], {}
        
    def load(self) -> Tuple[List[Document], Dict[str, Any]]:
        if self.async_mode:
            return asyncio.run(self.load_async())
        else:
            return self.load_sync()

    def _calculate_stats(self, docs: List[Any], figure_response: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            logging.debug(f"Starting _calculate_stats with docs={docs}, figure_response={figure_response}")
            
            total_figures = sum(len(item.get("figures", [])) for item in figure_response)
            logging.debug(f"Calculated total_figures: {total_figures}")
            
            total_tables = sum(len(item.get("tables", [])) for item in figure_response)
            logging.debug(f"Calculated total_tables: {total_tables}")
            
            total_pages = sum(item.get("pages", 0) for item in figure_response)
            logging.debug(f"Calculated total_pages: {total_pages}")
            
            total_time = sum(item.get("time_in_millis", 0) for item in figure_response)
            logging.debug(f"Calculated total_time: {total_time}")

            print(f"Total figures: {total_figures}, Total tables: {total_tables}, Total pages: {total_pages}, Total time: {total_time}")

            stats = {
                "total_documents": len(docs),
                "total_figures": total_figures,
                "total_tables": total_tables,
                "total_pages": total_pages,
                "figures_extracted_in_seconds": round(total_time / 1000, 2),
            }
            
            logging.debug(f"Calculated stats: {stats}")
            return stats
            
        except Exception as e:
            logging.error(f"Error calculating stats: {e}")
            logging.error(f"Exception traceback: {traceback.format_exc()}")
            return {
                "total_documents": 0,
                "total_figures": 0,  
                "total_tables": 0,
                "total_pages": 0,
                "figures_extracted_in_seconds": 0
            }
        

def main():
    logging.basicConfig(level=logging.DEBUG)
    
    # Provide a sample PDF path and output directory
    pdf_path = '/Users/zehrakorkusuz/PaperRAG/data/APOE4.pdf'
    output_dir = '/Users/zehrakorkusuz/PaperRAG/data/figures'
    
    # Create an instance of DocumentLoader
    loader = DocumentLoader(pdf_path=pdf_path, output_dir=output_dir, async_mode=False)
    
    # Load documents synchronously
    documents, stats = loader.load_sync()
    
    # Print the results
    print(f"Documents: {documents}")
    print(f"Stats: {stats}")

async def main():
    import json
    logging.basicConfig(level=logging.DEBUG)
    
    # Provide a sample PDF path and output directory
    #pdf_path = '/Users/zehrakorkusuz/PaperRAG/data/Brain Imaging Anomaly Detection.pdf'
    pdf_path = "/Users/zehrakorkusuz/PaperRAG/data"
    output_dir = '/Users/zehrakorkusuz/PaperRAG/data/figures'
    
    # Create an instance of DocumentLoader
    loader = DocumentLoader(pdf_path=pdf_path, output_dir=output_dir, async_mode=True)
    
    # Load documents asynchronously
    documents, stats = await loader.load_async()

    # save documents to json
    with open("documents.json", "w") as f:
        json.dump([doc.dict() for doc in documents], f, indent=4)
    
    # Print the results
    print(f"Documents: {documents}")
    print(f"Stats: {stats}")

if __name__ == '__main__':
    asyncio.run(main())