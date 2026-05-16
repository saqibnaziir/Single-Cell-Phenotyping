"""
LLM Integration for Interpretable Cell Analysis Summaries
========================================================
"Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning"
Saqib Nazir, Ardhendu Behera — Edge Hill University, UK
ICPR 2026 | arXiv:2605.14717

Converts model predictions to human-readable biological interpretations.
Supports:
- OpenAI GPT-4/GPT-3.5 (requires API key)
- Local LLMs via HuggingFace (Llama, Mistral)
- Rule-based fallback (no API required, zero dependencies beyond numpy)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


# ==================== PROMPT FORMATTING ==================== #

def format_predictions_for_llm(
    cls_probs: np.ndarray,
    predicted_class: int,
    proteins: np.ndarray,
    protein_names: List[str],
    uncertainty: Optional[np.ndarray] = None
) -> Dict:
    """
    Convert model predictions to structured format for LLM
    
    Args:
        cls_probs: Class probabilities (num_classes,)
        predicted_class: Predicted class index
        proteins: Protein expression values (num_proteins,)
        protein_names: List of protein marker names
        uncertainty: Optional uncertainty estimates
    
    Returns:
        Formatted dictionary for LLM processing
    """
    # Class mapping
    class_info = {
        0: {
            'name': 'Lymphocyte',
            'description': 'Immune defense cells including T-cells, B-cells, and NK cells'
        },
        1: {
            'name': 'Granulocyte',
            'description': 'Innate immunity cells including neutrophils, eosinophils, and basophils'
        },
        2: {
            'name': 'Monocyte',
            'description': 'Phagocytic cells that differentiate into macrophages and dendritic cells'
        }
    }
    
    class_data = class_info[predicted_class]
    confidence = cls_probs[predicted_class]
    
    # Top proteins
    top_protein_indices = np.argsort(np.abs(proteins))[::-1][:5]
    top_proteins = [
        {
            'name': protein_names[i],
            'level': float(proteins[i]),
            'abbreviation': protein_names[i].split('_')[0] if '_' in protein_names[i] else protein_names[i]
        }
        for i in top_protein_indices
    ]
    
    # All proteins formatted
    all_proteins = {
        name.split('_')[0] if '_' in name else name: float(val)
        for name, val in zip(protein_names, proteins)
    }
    
    formatted = {
        'cell_type': class_data['name'],
        'cell_type_description': class_data['description'],
        'confidence': float(confidence),
        'confidence_percent': f"{confidence*100:.1f}%",
        'top_proteins': top_proteins,
        'all_proteins': all_proteins,
        'has_uncertainty': uncertainty is not None
    }
    
    if uncertainty is not None:
        formatted['uncertainty'] = {
            'classification': float(uncertainty[0]) if len(uncertainty) > 0 else 0.0,
            'protein_average': float(uncertainty[1]) if len(uncertainty) > 1 else 0.0
        }
    
    return formatted


def build_llm_prompt(formatted_preds: Dict) -> str:
    """
    Build LLM prompt from formatted predictions
    
    Args:
        formatted_preds: Formatted prediction dictionary
    
    Returns:
        Complete prompt string
    """
    prompt = f"""CELL TYPE: {formatted_preds['cell_type']} ({formatted_preds['cell_type_description']})
CONFIDENCE: {formatted_preds['confidence_percent']}

TOP PROTEIN MARKERS:
"""
    
    # Add top proteins
    for i, protein in enumerate(formatted_preds['top_proteins'], 1):
        prompt += f"{i}. {protein['name']}: {protein['level']:.1f}\n"
    
    prompt += f"""
ALL PROTEIN EXPRESSION LEVELS:
"""
    
    # Add all proteins (abbreviated)
    for name, val in list(formatted_preds['all_proteins'].items())[:10]:
        prompt += f"- {name}: {val:.1f}\n"
    
    if formatted_preds.get('has_uncertainty'):
        prompt += f"\nUNCERTAINTY: Classification={formatted_preds['uncertainty']['classification']:.3f}, "
        prompt += f"Proteins={formatted_preds['uncertainty']['protein_average']:.3f}\n"
    
    prompt += """

TASK: Generate a concise, clinically relevant interpretation of this cell's functional state. Include:
1. What this cell type typically does in the immune system
2. What the protein expression pattern suggests about cell activation/function
3. Any notable protein combinations or absences
4. Clinical significance (if any)

Keep the response under 150 words and use accessible language for both clinical and research audiences.
"""

    return prompt


# ==================== LLM INTERFACES ==================== #

class LLMInterface:
    """Base class for LLM interfaces"""
    
    def __init__(self):
        pass
    
    def generate_interpretation(self, prompt: str) -> str:
        """Generate interpretation from prompt"""
        raise NotImplementedError


class OpenAIInterface(LLMInterface):
    """OpenAI GPT-4/3.5 interface"""
    
    def __init__(self, model: str = 'gpt-4', api_key: Optional[str] = None):
        """
        Args:
            model: Model name ('gpt-4', 'gpt-3.5-turbo', etc.)
            api_key: OpenAI API key (if None, uses environment variable)
        """
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def generate_interpretation(self, prompt: str) -> str:
        """Generate using OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': 'You are an expert immunologist.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content


class OpenRouterInterface(LLMInterface):
    """OpenRouter interface for various LLMs (including Kimi)"""
    
    def __init__(self, model: str = 'moonshot-v1-8k', api_key: Optional[str] = None):
        """
        Args:
            model: Model name ('moonshot-v1-8k', 'claude-3-haiku', etc.)
            api_key: OpenRouter API key
        """
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            self.model = model
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def generate_interpretation(self, prompt: str) -> str:
        """Generate using OpenRouter API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': 'You are an expert immunologist.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content


class GeminiInterface(LLMInterface):
    """Google Gemini interface (direct API)"""
    
    def __init__(self, model: str = 'gemini-2.0-flash', api_key: Optional[str] = None):
        """
        Args:
            model: Model name ('gemini-2.0-flash', 'gemini-2.5-flash', etc.)
            api_key: Google AI API key
        """
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=api_key)
            self.model_name = model
            self.model = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
    
    def generate_interpretation(self, prompt: str) -> str:
        """Generate using Google Gemini API"""
        # Add system instruction to prompt for Gemini
        full_prompt = "You are an expert immunologist analyzing single-cell protein expression data from label-free microscopy.\n\n" + prompt
        
        response = self.model.generate_content(
            full_prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 300
            }
        )
        return response.text


class LocalLLMInterface(LLMInterface):
    """Local LLM interface (Llama, Mistral, etc.)"""
    
    def __init__(self, model_name: str = 'meta-llama/Llama-2-7b-chat-hf'):
        """
        Args:
            model_name: HuggingFace model name
        """
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device_map="auto"
            )
        except ImportError:
            raise ImportError("transformers package not installed. Install with: pip install transformers")
    
    def generate_interpretation(self, prompt: str) -> str:
        """Generate using local LLM"""
        result = self.pipeline(prompt, max_length=400, temperature=0.7, do_sample=True)
        return result[0]['generated_text'].replace(prompt, '').strip()


class RuleBasedInterface(LLMInterface):
    """Rule-based interpretation (no API required)"""
    
    def __init__(self):
        # Define interpretation rules
        self.protein_roles = {
            'CD45': 'leukocyte common antigen',
            'CD3': 'T-cell marker',
            'CD14': 'monocyte/macrophage marker',
            'CD16': 'Fc receptor, neutrophil/monocyte activation',
            'CD19': 'B-cell marker',
            'CD56': 'NK cell marker',
            'CD123': 'IL-3 receptor',
            'HLADR': 'antigen presentation',
            'HLA-DR': 'antigen presentation',
            'autofluor': 'autofluorescence background'
        }
    
    def generate_interpretation(self, prompt: str) -> str:
        """Generate rule-based interpretation"""
        # Parse prompt (simple extraction)
        lines = prompt.split('\n')
        cell_type = None
        top_proteins = []
        
        for i, line in enumerate(lines):
            if 'CELL TYPE:' in line:
                cell_type = line.split(':')[1].split('(')[0].strip()
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                parts = line.split(':')
                if len(parts) == 2:
                    protein_name = parts[0].split('.')[1].strip()
                    protein_value = float(parts[1].strip())
                    top_proteins.append((protein_name, protein_value))
        
        # Generate interpretation
        interpretation = f"This is a {cell_type.lower()}. "
        
        # Add protein-specific insights
        if top_proteins:
            dominant_protein = top_proteins[0]
            protein_key = dominant_protein[0].split('_')[0] if '_' in dominant_protein[0] else dominant_protein[0]
            
            if protein_key in self.protein_roles:
                interpretation += f"High expression of {protein_key} ({dominant_protein[1]:.1f}) indicates {self.protein_roles[protein_key]}. "
        
        # Add cell type specific notes
        if cell_type == 'Granulocyte':
            interpretation += "High CD16 expression suggests mature neutrophil subtype with enhanced phagocytic and antibody-dependent cellular cytotoxicity capabilities. "
        elif cell_type == 'Monocyte':
            interpretation += "Monocytes are precursors to macrophages and dendritic cells, playing a key role in antigen presentation and innate immunity. "
        elif cell_type == 'Lymphocyte':
            interpretation += "Lymphocytes coordinate adaptive immune responses through cell-mediated and humoral immunity. "
        
        interpretation += "The protein expression pattern reflects functional specialization and activation state."
        
        return interpretation


# ==================== COMPLETE SUMMARIZER ==================== #

class CellAnalysisSummarizer:
    """Complete summarizer integrating model predictions and LLM"""
    
    def __init__(
        self,
        llm_interface: LLMInterface,
        protein_names: List[str],
        class_names: List[str]
    ):
        """
        Args:
            llm_interface: LLM interface (OpenAI, Local, or Rule-based)
            protein_names: List of protein marker names
            class_names: List of class names
        """
        self.llm_interface = llm_interface
        self.protein_names = protein_names
        self.class_names = class_names
    
    def summarize(
        self,
        cls_probs: np.ndarray,
        predicted_class: int,
        proteins: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Generate complete analysis summary
        
        Args:
            cls_probs: Class probabilities
            predicted_class: Predicted class index
            proteins: Protein expression values
            uncertainty: Uncertainty estimates
            metadata: Additional metadata
        
        Returns:
            Complete summary dictionary
        """
        # Format predictions
        formatted_preds = format_predictions_for_llm(
            cls_probs, predicted_class, proteins,
            self.protein_names, uncertainty
        )
        
        # Build prompt
        prompt = build_llm_prompt(formatted_preds)
        
        # Generate interpretation
        interpretation = self.llm_interface.generate_interpretation(prompt)
        
        # Compile summary
        summary = {
            'predictions': {
                'cell_type': formatted_preds['cell_type'],
                'confidence': formatted_preds['confidence_percent'],
                'protein_expression': formatted_preds['all_proteins'],
                'top_markers': formatted_preds['top_proteins']
            },
            'interpretation': interpretation,
            'uncertainty': formatted_preds.get('uncertainty'),
            'metadata': metadata or {}
        }
        
        return summary
    
    def save_summary(self, summary: Dict, output_path: Path, format: str = 'json'):
        """
        Save summary to file
        
        Args:
            summary: Summary dictionary
            output_path: Output file path
            format: Format ('json', 'txt', 'markdown')
        """
        output_path = Path(output_path)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        
        elif format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("CELL ANALYSIS SUMMARY\n")
                f.write("="*70 + "\n\n")
                f.write(f"Cell Type: {summary['predictions']['cell_type']}\n")
                f.write(f"Confidence: {summary['predictions']['confidence']}\n")
                f.write("\nTop Protein Markers:\n")
                for protein in summary['predictions']['top_markers']:
                    f.write(f"  - {protein['name']}: {protein['level']:.1f}\n")
                f.write("\nInterpretation:\n")
                f.write(summary['interpretation'] + "\n")
                f.write("="*70 + "\n")
        
        elif format == 'markdown':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Cell Analysis Summary\n\n")
                f.write(f"**Cell Type:** {summary['predictions']['cell_type']}  \n")
                f.write(f"**Confidence:** {summary['predictions']['confidence']}  \n\n")
                f.write("## Top Protein Markers\n\n")
                for protein in summary['predictions']['top_markers']:
                    f.write(f"- **{protein['name']}**: {protein['level']:.1f}\n")
                f.write("\n## Interpretation\n\n")
                f.write(summary['interpretation'] + "\n")
        
        print(f"✅ Summary saved to {output_path}")


# ==================== EXAMPLE USAGE ==================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate LLM Summaries for Cell Analysis')
    parser.add_argument('--llm_type', type=str, default='rule',
                       choices=['openai', 'openrouter', 'gemini', 'local', 'rule'],
                       help='LLM interface type')
    parser.add_argument('--model', type=str, default='gpt-4',
                       help='Model name (for OpenAI or local LLM)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key (OpenAI or environment variable)')
    parser.add_argument('--output', type=str, default='cell_analysis_summary.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Protein and class names
    protein_names = [
        # Full model unmixed (have 100% data in BSCCMNIST)
        'CD123/HLA-DR/CD14_full_model_unmixed',
        'CD3/CD19/CD56_full_model_unmixed',
        'CD45_full_model_unmixed',
        'CD16_full_model_unmixed'
    ]
    class_names = ['Lymphocyte', 'Granulocyte', 'Monocyte']
    
    # Create LLM interface
    if args.llm_type == 'openai':
        llm = OpenAIInterface(model=args.model, api_key=args.api_key)
    elif args.llm_type == 'openrouter':
        llm = OpenRouterInterface(model=args.model, api_key=args.api_key)
    elif args.llm_type == 'gemini':
        llm = GeminiInterface(model=args.model, api_key=args.api_key)
    elif args.llm_type == 'local':
        llm = LocalLLMInterface(model_name=args.model)
    else:
        llm = RuleBasedInterface()
    
    # Create summarizer
    summarizer = CellAnalysisSummarizer(llm, protein_names, class_names)
    
    # Example predictions
    print("Generating example interpretation...")
    
    example_cls_probs = np.array([0.05, 0.90, 0.05])  # High confidence Granulocyte
    example_proteins = np.array([65.3, 7.2, 0.9, 2.1, 1.5, 8.5, 45.8, 12.1, 1.8, 5.0])
    
    summary = summarizer.summarize(
        cls_probs=example_cls_probs,
        predicted_class=1,
        proteins=example_proteins,
        metadata={'sample_id': 'example_001'}
    )
    
    # Save summary
    output_path = Path(args.output)
    summarizer.save_summary(summary, output_path, format='json')
    
    # Print to console
    print("\n" + "="*70)
    print("EXAMPLE INTERPRETATION")
    print("="*70)
    print(summary['interpretation'])
    print("="*70)

