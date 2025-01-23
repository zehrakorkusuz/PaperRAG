from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

## Default initialization still works as before:
# doc_converter = DocumentConverter()

# previous `PipelineOptions` is now `PdfPipelineOptions`
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True
#...

## Custom options are now defined per format.
doc_converter = (
    DocumentConverter(  # all of the below is optional, has internal defaults.
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
        ],  # whitelist formats, non-matching files are ignored.
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, # pipeline options go here.
                backend=PyPdfiumDocumentBackend # optional: pick an alternative backend
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline # default for office formats and HTML
            ),
        },
    )
)