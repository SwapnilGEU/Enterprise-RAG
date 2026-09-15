"""Enterprise RAG pipeline — modularised from Prototype4.ipynb.

Import order (each module only imports from ones above it):

    config -> retry -> models -> extraction -> markdown -> sections
           -> chunking -> payload -> vector_store -> retrieval -> generation
           -> history -> agent
"""
