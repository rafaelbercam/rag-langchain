
📋 PROMPT PARA IMPLEMENTAÇÃO - APLICAÇÃO RAG COM LANGCHAIN

# Projeto: Implementar Aplicação RAG Completa baseada no Artigo "Guia Prático: Implementar RAG com LangChain"

## Objetivo
Criar uma aplicação Python profissional que implementa o sistema RAG descrito no artigo
`docs/blog/rag/2026-08-22-guia-rag-langchain.md`. A aplicação deve ser modular,
testável e documentada.

## Contexto do Artigo
O artigo apresenta um guia passo a passo para:
1. Carregar e processar documentos
2. Gerar embeddings com OpenAI
3. Armazenar vetores em FAISS
4. Criar chains de retrieval-QA
5. Implementar melhorias (reranking, query reformulation, memory conversacional)

## Estrutura Esperada do Projeto

src/
├── rag_app/
│   ├── init.py
│   ├── config.py              # Configurações (OPENAI_API_KEY, paths, etc)
│   ├── document_processor.py   # Carregamento e processamento de docs
│   ├── embeddings_manager.py   # Gestão de embeddings e vector store
│   ├── retriever.py            # Configuração de retrievers
│   ├── chains.py               # Chains RAG (básico e avançado)
│   ├── utils.py                # Utilitários gerais
│   └── cli.py                  # Interface CLI
├── tests/
│   ├── init.py
│   ├── test_document_processor.py
│   ├── test_embeddings.py
│   ├── test_retriever.py
│   ├── test_chains.py
│   └── integration_tests.py
├── data/
│   ├── sample_documents/       # Documentos de exemplo para teste
│   └── vectorstore/            # Armazenamento local do FAISS
├── notebooks/
│   └── demo.ipynb              # Jupyter notebook com demonstração
├── requirements.txt
├── .env.example
├── README.md
└── blog.md                      # Arquivo de documentação do processo


## Etapas de Implementação

### ETAPA 1: Setup e Configuração (config.py)
**Responsabilidades:**
- Carregar variáveis de ambiente (.env)
- Validar presença de OPENAI_API_KEY
- Definir constantes: CHUNK_SIZE (1000), CHUNK_OVERLAP (200), K_DOCS (3)
- Configurar paths para data e vector store
- Definir modelos: embedding_model="text-embedding-3-large", llm_model="gpt-3.5-turbo"

**Requisitos:**
- Usar `python-dotenv` para carregar .env
- Lançar exceção clara se OPENAI_API_KEY não existir
- Permitir sobrescrita de defaults via variáveis de ambiente

**Teste:**
- Validar que todas as constantes estão definidas corretamente
- Verificar path de diretórios críticos

---

### ETAPA 2: Processamento de Documentos (document_processor.py)
**Responsabilidades:**
- Carregar documentos de múltiplos formatos (.txt, .md, .pdf opcionalmente)
- Dividir em chunks usando CharacterTextSplitter
- Adicionar metadata (source, timestamp)
- Retornar lista de documentos processados

**Funcionalidades:**
```python
class DocumentProcessor:
    def load_documents(file_paths: List[str]) -> List[Document]
    def split_documents(documents: List[Document]) -> List[Document]
    def process(file_paths: List[str]) -> List[Document]

Requisitos:
- chunk_size=1000, chunk_overlap=200 (conforme artigo)
- Logging de quantidade de chunks criados
- Tratamento de erros para arquivos não encontrados
- Preserve informações de source nos metadados

Teste:
- Carregar arquivo de teste, verificar quantidade de chunks
- Validar que chunk_overlap funciona (última sentença do chunk N aparece no N+1)

---

ETAPA 3: Gestão de Embeddings (embeddings_manager.py)

Responsabilidades:
- Inicializar OpenAIEmbeddings
- Criar vector store FAISS a partir de documentos
- Salvar/carregar vector store do disco
- Executar buscas de similaridade

Funcionalidades:
class EmbeddingsManager:
    def create_vectorstore(documents: List[Document]) -> FAISS
    def save_vectorstore(vectorstore: FAISS, path: str) -> None
    def load_vectorstore(path: str) -> FAISS
    def search_similar(vectorstore: FAISS, query: str, k: int) -> List[Document]

Requisitos:
- Usar model="text-embedding-3-large" (conforme artigo)
- Criar diretório automaticamente se não existir
- Permitir recarregar de vectorstore já existente
- Retornar documentos com score de similaridade

Teste:
- Criar vectorstore, salvar, recarregar e validar integridade
- Testar search com queries conhecidas

---

ETAPA 4: Configuração de Retrievers (retriever.py)

Responsabilidades:
- Configurar retriever básico com k=3 (top-3 documentos)
- Implementar MultiQueryRetriever para reformulação de queries
- Implementar ContextualCompressionRetriever com reranking (opcional)
- Suportar diferentes estratégias de busca

Funcionalidades:
class RetrieverFactory:
    def create_basic_retriever(vectorstore: FAISS, k: int = 3) -> Retriever
    def create_multiquery_retriever(vectorstore: FAISS, llm) -> Retriever
    def create_compression_retriever(vectorstore: FAISS, llm, k: int = 3) -> Retriever

Requisitos:
- Retriever básico: search_type="similarity", search_kwargs={"k": 3}
- MultiQueryRetriever: gera múltiplas reformulações da query
- Logging de documentos recuperados

Teste:
- Testar cada tipo de retriever com queries de teste
- Validar que cada retriever retorna documentos relevantes

---

ETAPA 5: Chains RAG (chains.py)

Responsabilidades:
- Implementar RetrievalQA chain (básica)
- Implementar ConversationalRetrievalChain (com memory)
- Configurar LLM com temperature=0 (conforme artigo)
- Estruturar chains para retornar source_documents

Funcionalidades:
class RAGChainFactory:
    def create_qa_chain(retriever: Retriever, llm: LLM) -> RetrievalQA
    def create_conversational_chain(retriever: Retriever, llm: LLM) -> ConversationalRetrievalChain

Requisitos:
- LLM: gpt-3.5-turbo, temperature=0
- Chain type: "stuff" (conforme artigo)
- Retornar sempre source_documents para rastreabilidade
- Adicionar constraint no prompt: "Responda apenas baseado no contexto"

Teste:
- Testar query conhecida e validar resposta baseada no contexto
- Validar que source_documents são retornados

---

ETAPA 6: Interface CLI (cli.py)

Responsabilidades:
- Fornecer comandos interativos
- Inicializar vectorstore a partir de documentos
- Executar queries contra RAG
- Modo conversacional com memory
- Debug: visualizar documentos recuperados

Comandos esperados:
python -m rag_app.cli init --docs-path ./data/sample_documents
python -m rag_app.cli query "Como funciona RAG?"
python -m rag_app.cli chat (modo conversacional)
python -m rag_app.cli search "termo" (apenas retriever, sem LLM)

Requisitos:
- Usar click ou argparse para CLI
- Validar vectorstore antes de executar queries
- Exibir tempo de execução e custo estimado
- Logging detalhado

---

ETAPA 7: Testes Automatizados (tests/)

Casos de Teste Obrigatórios:

1. test_document_processor.py
   - Carregar documento, validar chunks criados
   - Validar chunk_overlap
   - Validar tratamento de erros para arquivo não encontrado
2. test_embeddings.py
   - Criar vectorstore a partir de documentos
   - Salvar e recarregar vectorstore
   - Validar que buscas retornam documentos similares
3. test_retriever.py
   - Cada tipo de retriever retorna k documentos
   - Documentos têm score de relevância
4. test_chains.py
   - RetrievalQA retorna 'result' e 'source_documents'
   - Response não contém alucinações óbvias
   - ConversationalChain mantém histórico
5. integration_tests.py
   - Fluxo completo: load → embed → query → retrieve → generate
   - CLI commands funcionam end-to-end

Requisitos:
- Usar pytest
- Usar documentos mock/fixture para testes rápidos
- Não fazer chamadas reais à OpenAI em testes unitários (mock)
- Coverage mínimo: 80%

---

ETAPA 8: Documentação do Processo (blog.md)

Este arquivo deve conter:

1. Titulo e Data
   - Projeto: Implementação RAG com LangChain
   - Data: [data de conclusão]
2. Resumo Executivo (3-4 linhas)
   - O que foi implementado
   - Tecnologias principais
   - Resultado final
3. Contexto
   - Link para artigo original
   - Motivação para criar a aplicação
   - Decisões arquiteturais
4. Fluxo Step-by-Step de Implementação
   - Para cada etapa (config, doc processor, embeddings, etc):
     - O que foi feito: descrição breve
     - Por quê: justificativa das escolhas
     - Desafios encontrados: problemas e soluções
     - Comando/código relevante: snippet importante
     - Tempo estimado: quanto levou
5. Estrutura de Arquivos Final
   - Árvore de diretórios
   - Descrição rápida de cada módulo
6. Como Usar a Aplicação
   - Instalação de dependências
   - Configuração de .env
   - Exemplos de uso básico
   - Exemplos de uso avançado (reranking, conversational)
7. Resultados e Aprendizados
   - Performance (tempo médio de query)
   - Custo estimado (baseado em chamadas reais)
   - Qualidade de respostas (exemplos)
   - O que funcionou bem
   - O que pode melhorar
8. Próximos Passos
   - Melhorias não implementadas
   - Alternativas de vector stores (Pinecone, Weaviate)
   - Otimizações de custo
   - Integrações sugeridas
9. Recursos Utilizados
   - Links para documentação
   - Ferramentas e bibliotecas
   - Referências úteis
10. Conclusão
    - Reflexão final
    - Tempo total do projeto
    - Recomendações para reproduzir

---

Arquivos de Entrada Esperados

.env.example

OPENAI_API_KEY=sk-...
ENVIRONMENT=development
LOG_LEVEL=INFO

requirements.txt

langchain==0.1.x
openai==1.x
faiss-cpu==1.7.x
python-dotenv==1.0.x
click==8.x (ou argparse)
pytest==7.x
pytest-cov==4.x

data/sample_documents/example.txt

- Documento de teste com 5-10 parágrafos

---

Critérios de Aceitação

✅ Código:
- Todos os 8 módulos implementados e importáveis
- 80%+ coverage de testes automatizados
- Sem erros ao executar python -m rag_app.cli query "teste"
- Type hints em todos os arquivos

✅ Funcionalidade:
- Criar vectorstore de documentos
- Recuperar documentos relevantes com scores
- Gerar respostas usando RetrievalQA
- Executar queries conversacionais com memory

✅ Documentação:
- README.md com instruções claras
- blog.md com processo step-by-step completo
- Docstrings em classes/funções principais
- Comentários em pontos não-óbvios

✅ Performance:
- Query típica executa em < 5 segundos
- Memory conversacional mantém histórico

---

Entrega Final

Estrutura esperada após conclusão:
/seu-projeto-rag/
├── src/rag_app/          ✅ Todos 7 módulos
├── tests/                ✅ 5+ testes passando
├── data/                 ✅ Sample documents
├── notebooks/            ✅ demo.ipynb
├── requirements.txt      ✅
├── .env.example          ✅
├── README.md             ✅
└── blog.md               ✅ DOCUMENTAÇÃO DO PROCESSO


Arquivo blog.md deve ser criado/atualizado com o processo completo.

---

Notas Importantes

1. Reutilize do artigo: Pegue snippets de código do artigo, mas refatore para modularidade
2. Constrains do contexto: "Responda apenas baseado no contexto" deve estar no system prompt
3. Logging: Use logging nativo Python em todos os módulos
4. Tratamento de erros: Mensagens claras para: arquivo não encontrado, API falha, vectorstore inválido
5. Modo debug: CLI deve ter flag --verbose para debug detalhado

