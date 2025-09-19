from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
from dataclasses import dataclass
import sqlite3
import hashlib
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic Models
class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    context: Optional[str] = Field(None, description="Additional context about the user's farm/location")
    crop_type: Optional[str] = Field(None, description="Specific crop user is asking about")

class ChatResponse(BaseModel):
    answer: str
    confidence_score: float
    sources: List[str]
    related_topics: List[str]
    response_type: str  # "rag", "llm", "hybrid"

@dataclass
class KnowledgeDocument:
    id: str
    content: str
    title: str
    category: str
    metadata: Dict[str, Any]

class AgricultureRAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.knowledge_base: List[KnowledgeDocument] = []
        self.db_path = "agriculture_kb.db"
        self.embedding_dim = 384
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the RAG system components"""
        try:
            # Load sentence transformer model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize vector store
            self.vector_store = faiss.IndexFlatIP(self.embedding_dim)
            
            # Initialize database
            self.init_database()
            
            # Load knowledge base
            self.load_knowledge_base()
            
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def init_database(self):
        """Initialize SQLite database for storing documents and metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_knowledge_base(self):
        """Load comprehensive agriculture knowledge base"""
        agriculture_knowledge = [
        # WATER MANAGEMENT & IRRIGATION
        {
            "title": "Advanced Drip Irrigation Systems",
            "content": """Drip irrigation delivers water directly to plant roots through a network of tubing, emitters, and valves.
            System components include main lines, sub-main lines, laterals, emitters, filters, and pressure regulators.
            Proper filtration prevents clogging - screen filters for organic matter, sand filters for fine particles.
            Emitter flow rates typically range from 0.5-4.0 GPH. Spacing depends on soil type and crop requirements.
            Automation with timers and soil moisture sensors can optimize water use efficiency by 40-60%.
            Regular maintenance includes flushing lines, checking emitters, and replacing worn components.
            Cost analysis shows ROI within 2-3 years for most commercial operations.""",
            "category": "irrigation"
        },
        {
            "title": "Sprinkler Irrigation Design and Management",
            "content": """Sprinkler systems include overhead, center pivot, and linear move systems. Design considerations include
            pressure requirements, nozzle selection, spacing patterns, and wind drift management.
            Uniformity coefficient should be above 85% for optimal performance. Check with catch cans placed in grid pattern.
            Application rates must not exceed soil infiltration rates to prevent runoff and erosion.
            Center pivot systems cover 125-135 acres with 95% efficiency when properly managed.
            Variable rate irrigation (VRI) technology allows site-specific water application based on soil and crop needs.
            Energy costs can be reduced 20-30% with proper pump sizing and off-peak electricity scheduling.""",
            "category": "irrigation"
        },
        {
            "title": "Water Quality and Treatment",
            "content": """Water quality affects crop health, soil conditions, and irrigation system performance.
            Key parameters include pH (6.0-8.5 ideal), electrical conductivity, sodium adsorption ratio (SAR), and mineral content.
            High salinity water (EC > 3.0 dS/m) requires careful management and salt-tolerant crops.
            Iron and manganese levels above 0.3 ppm can cause staining and system clogging.
            Biological contamination requires chlorination or UV treatment for food safety compliance.
            Water testing should be done annually or when sources change. Treatment options include acidification,
            filtration, and chemical injection systems.""",
            "category": "irrigation"
        },
        {
            "title": "Deficit Irrigation Strategies",
            "content": """Deficit irrigation applies less water than full crop requirements to maximize water use efficiency.
            Regulated deficit irrigation (RDI) reduces water during less sensitive growth stages.
            Critical periods to maintain full irrigation: germination, flowering, fruit set, and grain filling.
            Stress indicators include leaf wilting, reduced leaf area, and changed canopy temperature.
            Partial root zone drying alternates irrigation between root zones to trigger stress responses.
            Economic analysis often shows better profit per unit water with moderate deficit irrigation.
            Crop-specific strategies vary - stone fruits benefit from late-season deficit, grains from mid-season management.""",
            "category": "irrigation"
        },

        # SOIL MANAGEMENT
        {
            "title": "Soil Structure and Aggregate Stability",
            "content": """Soil structure refers to arrangement of soil particles into aggregates or peds.
            Good structure improves water infiltration, root penetration, and gas exchange.
            Aggregate stability is measured by wet sieving - stable aggregates indicate healthy soil biology.
            Bulk density should be <1.4 g/cm³ for sandy soils, <1.1 g/cm³ for clay soils.
            Compaction from machinery reduces pore space and limits root growth. Use controlled traffic farming.
            Biological agents like mycorrhizal fungi and earthworms create stable aggregates.
            Cover crops, reduced tillage, and organic amendments improve structure over time.""",
            "category": "soil_management"
        },
        {
            "title": "Advanced Soil Testing and Nutrient Management",
            "content": """Comprehensive soil testing includes pH, organic matter, CEC, base saturation, and micronutrients.
            Grid sampling (2.5-acre grids) provides spatial variability data for precision agriculture.
            Soil pH affects nutrient availability - phosphorus peaks at 6.5, most nutrients available at 6.0-7.0.
            Cation exchange capacity (CEC) indicates soil's ability to hold nutrients. Clay soils have higher CEC.
            Base saturation ratios: Calcium 65-75%, Magnesium 10-15%, Potassium 2-5%, Sodium <3%.
            Tissue testing during growing season provides real-time nutrient status.
            Variable rate application based on soil test maps can reduce fertilizer costs 15-25%.""",
            "category": "soil_management"
        },
        {
            "title": "Soil Biology and Microbial Health",
            "content": """Soil microorganisms include bacteria, fungi, protozoa, nematodes, and arthropods.
            Bacterial:fungal ratio indicates soil health - 1:1 for annual crops, up to 1:100 for forests.
            Beneficial bacteria fix nitrogen, solubilize phosphorus, and produce plant growth hormones.
            Mycorrhizal fungi extend root systems 100-1000 times, improving nutrient and water uptake.
            Soil respiration indicates biological activity - higher rates suggest active microbial communities.
            Pesticides and excessive tillage can harm soil biology. Use targeted applications when necessary.
            Compost, biochar, and microbial inoculants can restore biological activity.""",
            "category": "soil_management"
        },
        {
            "title": "Soil Conservation and Erosion Control",
            "content": """Soil erosion removes 10-40 tons per acre annually on sloping land without conservation practices.
            Water erosion types include sheet, rill, and gully erosion. Wind erosion affects sandy, dry soils.
            Conservation tillage maintains 30% residue cover to reduce erosion by 50-90%.
            Contour farming, strip cropping, and terraces slow water flow on slopes.
            Buffer strips and waterways protect streams from sediment and chemical runoff.
            Cover crops reduce erosion, add organic matter, and suppress weeds.
            Economic losses from erosion include reduced productivity and off-site environmental damage.""",
            "category": "soil_management"
        },

        # PEST MANAGEMENT
        {
            "title": "Beneficial Insect Conservation and Habitat Management",
            "content": """Beneficial insects provide $4.5 billion in pest control services annually in US agriculture.
            Key beneficials include parasitic wasps, predatory beetles, lacewings, and hover flies.
            Habitat requirements include nectar sources, overwintering sites, and refuge areas.
            Plant diverse flowering plants: alyssum, sunflower, buckwheat, and native wildflowers.
            Insectary strips within fields provide season-long beneficial insect habitat.
            Minimize broad-spectrum insecticide use - choose selective products when treatment needed.
            Biological control releases include Trichogramma wasps for moth control.""",
            "category": "pest_control"
        },
        {
            "title": "Integrated Disease Management",
            "content": """Plant diseases cause 20-40% yield losses globally. Fungal diseases are most common.
            Disease triangle requires host, pathogen, and favorable environment for infection.
            Cultural controls include crop rotation, sanitation, resistant varieties, and plant spacing.
            Biological controls use beneficial microorganisms like Bacillus subtilis and Trichoderma.
            Fungicide resistance management requires rotating modes of action and tank mixing.
            Scouting protocols include regular field monitoring and weather-based disease models.
            Economic thresholds help determine when treatment costs are justified.""",
            "category": "pest_control"
        },
        {
            "title": "Weed Management Strategies",
            "content": """Weeds compete for water, nutrients, light, and space, reducing crop yields 10-50%.
            Integrated weed management combines cultural, mechanical, biological, and chemical methods.
            Crop rotation disrupts weed life cycles and allows different herbicide modes of action.
            Cover crops suppress weeds through competition and allelopathic compounds.
            Herbicide resistance is increasing - rotate modes of action and use multiple tactics.
            Pre-emergence herbicides provide early season control, post-emergence for emerged weeds.
            Mechanical cultivation timing is critical - cultivate when weeds are small and soil conditions suitable.""",
            "category": "pest_control"
        },
        {
            "title": "Rodent and Vertebrate Pest Control",
            "content": """Rodents cause billions in crop damage annually through direct feeding and contamination.
            Integrated rodent management includes habitat modification, exclusion, and population control.
            Remove food sources, nesting sites, and shelter around crop areas and storage facilities.
            Trapping is effective for small populations - use snap traps or live traps as appropriate.
            Rodenticides should be used in bait stations to protect non-target species.
            Bird damage management includes netting, scare devices, and harvest timing adjustment.
            Wildlife corridors and buffer zones can redirect animals away from crop areas.""",
            "category": "pest_control"
        },

        # CROP NUTRITION
        {
            "title": "Precision Nutrient Management",
            "content": """Precision nutrient management matches fertilizer application to crop needs by location and time.
            Variable rate technology uses GPS and soil/yield maps to adjust application rates across fields.
            Soil sampling on 2.5-acre grids provides detailed nutrient variability information.
            Plant tissue testing during growing season indicates real-time nutrient status.
            Chlorophyll meters and NDVI sensors provide rapid assessment of nitrogen status.
            Enhanced efficiency fertilizers include slow-release, stabilized, and coated products.
            Economic analysis shows precision management can reduce fertilizer costs 10-20% while maintaining yields.""",
            "category": "nutrition"
        },
        {
            "title": "Organic Matter and Carbon Cycling",
            "content": """Soil organic matter contains 50-60% carbon and affects all soil properties.
            Active organic matter (1-5 years turnover) provides nutrients, stable organic matter improves structure.
            Carbon:nitrogen ratios affect decomposition - high C:N materials immobilize nitrogen temporarily.
            Composting process requires proper C:N ratio (25-30:1), moisture, oxygen, and temperature management.
            Vermicomposting uses earthworms to process organic materials into high-quality compost.
            Biochar provides long-term carbon storage and improves soil water and nutrient retention.
            Cover crops contribute 2-4 tons per acre of biomass, adding significant organic matter.""",
            "category": "nutrition"
        },
        {
            "title": "Micronutrient Management",
            "content": """Micronutrients required in small amounts but essential for crop health and quality.
            Deficiency symptoms: Iron - interveinal chlorosis; Zinc - stunted growth; Boron - hollow stems.
            Soil pH affects micronutrient availability - alkaline soils often deficient in iron, zinc, manganese.
            Foliar applications provide rapid correction of deficiencies during growing season.
            Chelated forms remain available longer but cost more than sulfate forms.
            Tissue testing is more reliable than soil testing for micronutrient status.
            Over-application can cause toxicity and interfere with uptake of other nutrients.""",
            "category": "nutrition"
        },

        # CROP-SPECIFIC GUIDANCE
        {
            "title": "Corn Production Management",
            "content": """Corn requires 1.2-1.4 inches water per week during grain filling period.
            Plant population of 32,000-36,000 plants per acre optimal for most conditions.
            Nitrogen requirements: 1.1-1.3 lbs N per bushel yield goal. Side-dress at V6-V8 stage.
            Critical growth stages: V6 (ear size determination), VT-R1 (pollination), R3-R5 (grain filling).
            Common pests: corn rootworm, European corn borer, army worm. Scout weekly during season.
            Diseases: gray leaf spot, northern corn leaf blight, common rust. Use resistant hybrids.
            Harvest at 20-25% moisture for maximum yield, dry to 15.5% for safe storage.""",
            "category": "crop_management"
        },
        {
            "title": "Soybean Production Systems",
            "content": """Soybeans fix 50-60% of nitrogen needs through Rhizobia bacteria symbiosis.
            Inoculation with proper rhizobia strain increases yields 5-10% on new ground.
            Plant population: 120,000-140,000 plants per acre in 7.5-30 inch rows.
            Water requirements peak during pod filling (R3-R6 stages) at 1.5 inches per week.
            Major pests: soybean aphid, spider mites, stink bugs. Economic thresholds vary by stage.
            Diseases: sudden death syndrome, white mold, frogeye leaf spot. Manage with rotation and resistance.
            Harvest when pods rattle and moisture is 13-15% for best quality and storage.""",
            "category": "crop_management"
        },
        {
            "title": "Wheat Production and Management",
            "content": """Winter wheat planted in fall, spring wheat planted early spring. Choose adapted varieties.
            Seeding rate: 1.2-1.6 million seeds per acre depending on variety and planting date.
            Nitrogen application: 30-50% at planting, remainder at spring green-up and boot stage.
            Growth stages: tillering, jointing, boot, heading, grain filling, maturity.
            Diseases: Fusarium head blight, stripe rust, powdery mildew. Fungicide timing critical.
            Pests: Hessian fly, armyworm, aphids. Monitor during heading and grain filling.
            Harvest at 13-14% moisture when kernels are hard and golden brown.""",
            "category": "crop_management"
        },
        {
            "title": "Vegetable Crop Intensive Management",
            "content": """Vegetable production requires intensive management for high value crops.
            Transplant production in greenhouses provides head start and uniform stands.
            Plasticulture systems use plastic mulch, drip irrigation, and row covers for season extension.
            Fertility programs require frequent, light applications due to shallow root systems.
            Pest monitoring weekly or bi-weekly due to rapid pest buildup potential.
            Harvest timing critical for quality - multiple harvests often required.
            Post-harvest handling includes rapid cooling, proper storage temperature and humidity.""",
            "category": "crop_management"
        },

        # CLIMATE AND WEATHER
        {
            "title": "Microclimate Management and Modification",
            "content": """Microclimates vary within fields due to topography, soil type, and vegetation.
            South-facing slopes warm earlier in spring, receive more solar radiation.
            Cold air drainage creates frost pockets in low areas - avoid planting sensitive crops there.
            Windbreaks reduce wind speed, modify temperature, and improve humidity.
            Row covers, high tunnels, and greenhouses extend growing seasons.
            Thermal mass from water, concrete, or stone moderates temperature fluctuations.
            Reflective mulches can reduce soil temperature and repel certain insects.""",
            "category": "climate"
        },
        {
            "title": "Growing Degree Day Systems",
            "content": """Growing degree days (GDD) predict plant development based on temperature accumulation.
            Base temperature varies by crop: corn 50°F, soybeans 50°F, wheat 32°F.
            Daily GDD = (Max temp + Min temp)/2 - Base temperature (if above base).
            Accumulate GDD from planting to predict emergence, flowering, and maturity dates.
            Corn requires 2700-3000 GDD for full maturity, varies by hybrid.
            GDD models help time pesticide applications, irrigation, and harvest operations.
            Climate change is shifting GDD accumulation patterns and growing zones.""",
            "category": "climate"
        },
        {
            "title": "Drought Management and Resilience",
            "content": """Drought is recurring challenge requiring proactive planning and adaptive management.
            Pre-season planning includes crop selection, soil moisture conservation, and water storage.
            Drought-tolerant varieties reduce yield losses during water-limited conditions.
            Soil management practices: maintain organic matter, reduce compaction, improve infiltration.
            Irrigation scheduling based on soil moisture monitoring and crop water use models.
            Crop insurance and risk management tools help protect against drought losses.
            Long-term strategies include diversification, water-efficient crops, and infrastructure improvements.""",
            "category": "climate"
        },

        # SUSTAINABLE AGRICULTURE
        {
            "title": "Regenerative Agriculture Principles",
            "content": """Regenerative agriculture focuses on rebuilding soil health and ecosystem function.
            Core principles: minimize soil disturbance, maximize crop diversity, keep living roots, integrate livestock.
            Cover crops are foundation - use diverse mixes to maximize soil biology benefits.
            No-till or minimal tillage preserves soil structure and organic matter.
            Rotational grazing improves soil carbon sequestration and pasture productivity.
            Ecosystem services include carbon storage, water filtration, and biodiversity conservation.
            Economic benefits develop over 3-7 years as soil health improves.""",
            "category": "sustainable_farming"
        },
        {
            "title": "Agroecology and Biodiversity",
            "content": """Agroecological systems mimic natural ecosystems while producing food and fiber.
            Biodiversity above and below ground supports ecosystem stability and resilience.
            Companion planting and polyculture systems maximize space and resource use.
            Native plant strips provide beneficial insect habitat and pollinator support.
            Integrated crop-livestock systems cycle nutrients and energy efficiently.
            Traditional ecological knowledge contributes to sustainable farming practices.
            Research shows diverse systems are more stable and profitable long-term.""",
            "category": "sustainable_farming"
        },

        # PRECISION AGRICULTURE
        {
            "title": "GPS and Auto-Guidance Systems",
            "content": """GPS technology enables precise field operations and reduces overlap/gaps.
            RTK GPS provides sub-inch accuracy for planting, spraying, and harvest operations.
            Auto-guidance systems reduce operator fatigue and improve operation precision.
            Controlled traffic farming uses permanent wheel tracks to reduce soil compaction.
            Field mapping creates accurate boundaries and area calculations for inputs.
            Integration with implements enables variable rate applications and section control.
            Return on investment typically achieved within 2-3 years through input savings.""",
            "category": "precision_agriculture"
        },
        {
            "title": "Remote Sensing and Crop Monitoring",
            "content": """Satellite and drone imagery provide regular crop monitoring capabilities.
            NDVI (Normalized Difference Vegetation Index) indicates crop health and stress.
            Thermal imagery detects water stress before visible symptoms appear.
            Multispectral sensors can identify nitrogen deficiency, disease, and pest damage.
            Time-series imagery tracks crop development and identifies problem areas.
            Machine learning algorithms analyze imagery to predict yields and optimize management.
            Integration with ground-truthing improves accuracy and interpretation.""",
            "category": "precision_agriculture"
        },
        {
            "title": "Data Management and Analytics",
            "content": """Farm data management systems integrate information from multiple sources.
            Yield monitoring during harvest creates detailed productivity maps.
            As-applied maps record actual inputs used for regulatory compliance and analysis.
            Weather data integration improves decision-making for field operations.
            Economic analysis tools evaluate profitability of different management zones.
            Cloud-based platforms enable data sharing with advisors and service providers.
            Data privacy and security considerations important for farm information protection.""",
            "category": "precision_agriculture"
        },

        # FARM BUSINESS MANAGEMENT
        {
            "title": "Enterprise Budgeting and Financial Analysis",
            "content": """Enterprise budgets estimate costs and returns for specific crop or livestock enterprises.
            Variable costs include seed, fertilizer, chemicals, fuel, and labor directly tied to production.
            Fixed costs include land rent, machinery depreciation, insurance, and facility costs.
            Break-even analysis determines minimum price or yield needed for profitability.
            Sensitivity analysis evaluates how changes in prices or yields affect profits.
            Cash flow projections help plan financing needs and payment timing.
            Benchmarking against similar operations identifies improvement opportunities.""",
            "category": "farm_management"
        },
        {
            "title": "Risk Management and Insurance",
            "content": """Agricultural risks include production, market, financial, legal, and human resource risks.
            Crop insurance protects against yield losses from weather, disease, and other covered causes.
            Revenue insurance products protect against both yield and price declines.
            Diversification across crops, markets, and enterprises reduces overall risk exposure.
            Marketing strategies include forward contracts, options, and cooperative marketing.
            Financial management includes maintaining adequate working capital and credit reserves.
            Succession planning ensures farm continuity across generations.""",
            "category": "farm_management"
        },

        # LIVESTOCK INTEGRATION
        {
            "title": "Rotational Grazing Systems",
            "content": """Rotational grazing improves pasture productivity and animal performance.
            Paddock systems allow rest periods for forage recovery and root development.
            Stocking density affects pasture utilization - higher density improves uniformity.
            Grazing height management: start grazing at 8-10 inches, move at 3-4 inches.
            Water systems and fencing infrastructure required for flexible management.
            Forage species selection includes grasses, legumes, and forbs for nutrition diversity.
            Integration with crop production through grazing crop residues and cover crops.""",
            "category": "livestock_integration"
        },
        {
            "title": "Manure Management and Nutrient Cycling",
            "content": """Livestock manure provides valuable nutrients but requires proper management.
            Manure nutrient content varies by species, diet, and bedding materials used.
            Storage systems must prevent runoff and groundwater contamination.
            Composting process stabilizes nutrients and reduces pathogen risks.
            Application timing should match crop nutrient needs and soil conditions.
            Calibrate spreaders to apply correct rates based on nutrient analysis.
            Records required for regulatory compliance and nutrient management planning.""",
            "category": "livestock_integration"
        },

        # ORGANIC CERTIFICATION AND PRODUCTION
        {
            "title": "Organic Certification Process and Standards",
            "content": """USDA Organic certification requires 3-year transition period without prohibited substances.
            Organic System Plan documents all practices, inputs, and monitoring procedures.
            Approved substances list specifies allowed fertilizers, pest control products, and additives.
            Record keeping requirements include all inputs, practices, and harvest/sales records.
            Annual inspections verify compliance with organic standards and practices.
            Certification maintains consumer confidence and access to premium markets.
            Violations can result in loss of certification and significant financial penalties.""",
            "category": "organic_production"
        },
        {
            "title": "Organic Soil Fertility Management",
            "content": """Organic systems rely on biological processes for nutrient cycling and availability.
            Compost applications provide slow-release nutrients and improve soil biology.
            Green manures and cover crops fix nitrogen and add organic matter.
            Rock minerals like rock phosphate provide long-term phosphorus availability.
            Microbial inoculants enhance nutrient cycling and plant health.
            Crop rotations with legumes maintain nitrogen fertility naturally.
            Soil testing and plant tissue analysis guide fertility management decisions.""",
            "category": "organic_production"
        },

        # POST-HARVEST AND STORAGE
        {
            "title": "Grain Storage and Quality Management",
            "content": """Proper grain storage maintains quality and prevents losses from insects, mold, and rodents.
            Moisture content must be below safe levels: corn 15.5%, soybeans 13%, wheat 13.5%.
            Aeration systems maintain uniform temperature and prevent moisture migration.
            Temperature monitoring detects hot spots that indicate pest or mold activity.
            Insect management includes sanitation, monitoring, and targeted treatments.
            Quality testing includes moisture, test weight, protein content, and damage assessment.
            Storage insurance protects against losses from fire, weather, and other perils.""",
            "category": "post_harvest"
        },
        {
            "title": "Fresh Produce Handling and Storage",
            "content": """Fresh produce requires rapid cooling to maintain quality and extend shelf life.
            Hydrocooling, forced air cooling, and vacuum cooling methods available.
            Storage temperature and humidity requirements vary by commodity.
            Ethylene management prevents premature ripening and quality degradation.
            Packaging protects produce and provides consumer information.
            Cold chain maintenance critical from harvest through retail sale.
            Food safety protocols include GAPs certification and traceability systems.""",
            "category": "post_harvest"
        },

        # EMERGING TECHNOLOGIES
        {
            "title": "Artificial Intelligence in Agriculture",
            "content": """AI applications in agriculture include image recognition, predictive analytics, and autonomous systems.
            Machine learning algorithms analyze satellite imagery to predict yields and detect problems.
            Computer vision systems identify weeds, pests, and diseases for targeted treatment.
            Robotic systems perform planting, weeding, and harvesting operations.
            Predictive models forecast weather impacts and optimize management decisions.
            Natural language processing enables farmer-friendly interfaces for complex systems.
            Edge computing allows real-time processing of field data without internet connectivity.""",
            "category": "emerging_technology"
        },
        {
            "title": "Vertical Farming and Controlled Environment Agriculture",
            "content": """Controlled environment agriculture produces crops in climate-controlled facilities.
            LED lighting systems provide specific light spectra optimized for plant growth.
            Hydroponic and aeroponic systems deliver nutrients directly to plant roots.
            Environmental controls manage temperature, humidity, CO2, and air circulation.
            Year-round production possible regardless of weather conditions.
            Water use efficiency 95% higher than field production through recirculation.
            High capital costs require premium markets and efficient production systems.""",
            "category": "emerging_technology"
        },

        # CLIMATE SMART AGRICULTURE
        {
            "title": "Carbon Sequestration and Climate Mitigation",
            "content": """Agriculture can sequester carbon in soil organic matter and biomass.
            No-till practices maintain soil carbon by reducing oxidation from tillage.
            Cover crops add carbon through photosynthesis and root biomass.
            Perennial crops and agroforestry systems store carbon long-term.
            Carbon credit markets provide potential revenue for sequestration practices.
            Measurement and verification protocols ensure real and additional benefits.
            Co-benefits include improved soil health, water quality, and biodiversity.""",
            "category": "climate_smart"
        },
        {
            "title": "Climate Adaptation Strategies",
            "content": """Climate change requires adaptive management strategies for agriculture.
            Variety selection focuses on heat tolerance, drought resistance, and disease resistance.
            Shifting planting dates optimizes growing conditions under changing climate.
            Infrastructure improvements include drainage, irrigation, and storage facilities.
            Diversification reduces risks from increased weather variability.
            Early warning systems help farmers prepare for extreme weather events.
            Research and extension support development and adoption of adaptive practices.""",
            "category": "climate_smart"
        }
    ]
        # Convert to KnowledgeDocument objects and store
        embeddings = []
        for idx, doc in enumerate(agriculture_knowledge):
            doc_id = hashlib.md5(doc["content"].encode()).hexdigest()
            knowledge_doc = KnowledgeDocument(
                id=doc_id,
                content=doc["content"],
                title=doc["title"],
                category=doc["category"],
                metadata={"source": "internal_kb", "index": idx}
            )
            self.knowledge_base.append(knowledge_doc)
            
            # Generate embeddings
            embedding = self.embedding_model.encode([doc["content"]])[0]
            embeddings.append(embedding)
        
        # Add embeddings to vector store
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
            self.vector_store.add(embeddings_array)
            
        logger.info(f"Loaded {len(self.knowledge_base)} documents into knowledge base")
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[tuple]:
        """Retrieve most relevant documents for a given query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search vector store
            scores, indices = self.vector_store.search(query_embedding, top_k)
            
            # Return documents with scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.knowledge_base):
                    results.append((self.knowledge_base[idx], float(score)))
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_llm_response(self, query: str, context_docs: List[KnowledgeDocument]) -> Dict[str, Any]:
        """Generate response using LLM with retrieved context"""
        try:
            # Prepare context from retrieved documents
            context = "\n\n".join([f"Document: {doc.title}\nContent: {doc.content}" for doc in context_docs])
            
            # Create prompt for LLM
            prompt = f"""
            You are an expert agricultural advisor. Based on the following knowledge and the user's question, 
            provide a comprehensive, accurate, and practical answer.
            
            Context Information:
            {context}
            
            User Question: {query}
            
            Instructions:
            - Provide specific, actionable advice
            - Include relevant technical details
            - Mention any important considerations or warnings
            - If the context doesn't fully answer the question, say so clearly
            - Keep the response focused and practical
            """
            
            # Simulate LLM response (replace with actual OpenAI API call)
            # For demo purposes, we'll use a rule-based response generator
            response = self.simulate_llm_response(query, context_docs)
            
            return {
                "answer": response,
                "confidence": 0.85,  # This would come from the LLM
                "sources": [doc.title for doc in context_docs]
            }
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing your query.",
                "confidence": 0.0,
                "sources": []
            }
    
    def simulate_llm_response(self, query: str, context_docs: List[KnowledgeDocument]) -> str:
        """Simulate LLM response based on retrieved context"""
        query_lower = query.lower()
        
        # Extract relevant information from context documents
        relevant_info = []
        for doc in context_docs:
            relevant_info.append(f"From {doc.title}: {doc.content[:200]}...")
        
        # Generate response based on query type
        if any(keyword in query_lower for keyword in ["water", "irrigation", "watering"]):
            return f"""Based on current agricultural best practices, here's what you need to know about watering:

{relevant_info[0] if relevant_info else 'Most crops require 1-1.5 inches of water per week, including rainfall.'}

Key recommendations:
• Monitor soil moisture at 6-8 inch depth
• Water deeply but less frequently to encourage root growth  
• Consider drip irrigation for water efficiency
• Adjust watering based on crop growth stage and weather conditions

Would you like specific advice for a particular crop or growing condition?"""
            
        elif any(keyword in query_lower for keyword in ["pest", "insect", "bug"]):
            return f"""For effective pest management, I recommend an Integrated Pest Management (IPM) approach:

{relevant_info[0] if relevant_info else 'IPM combines multiple strategies to control pests while minimizing environmental impact.'}

Action steps:
• Monitor weekly using visual inspections and sticky traps
• Encourage beneficial insects like ladybugs and parasitic wasps
• Use targeted treatments like neem oil for soft-bodied insects
• Practice crop rotation to break pest life cycles

What specific pest issues are you experiencing?"""
            
        elif any(keyword in query_lower for keyword in ["soil", "fertilizer", "nutrients"]):
            return f"""Soil health is fundamental to successful farming. Here's what you should know:

{relevant_info[0] if relevant_info else 'Healthy soil requires proper pH (6.0-7.5), adequate organic matter, and balanced nutrients.'}

Essential practices:
• Test soil every 2-3 years for pH and nutrients
• Add organic matter through compost or cover crops
• Balance N-P-K based on crop needs and soil test results
• Monitor soil structure and drainage

Do you have recent soil test results, or would you like guidance on soil testing?"""
            
        else:
            # General agricultural advice
            if relevant_info:
                return f"""Based on the available information: {relevant_info[0]}

This relates to your question about {query}. For the most accurate advice, I'd recommend:
• Consulting with your local agricultural extension office
• Considering your specific growing conditions and climate
• Testing any new practices on a small scale first

Could you provide more specific details about your farming situation for more targeted advice?"""
            else:
                return f"""Thank you for your question about {query}. While I don't have specific information readily available, 
here are some general recommendations:

• Consult your local agricultural extension service
• Connect with experienced farmers in your area  
• Consider soil and tissue testing for data-driven decisions
• Start with proven practices for your region and crop type

If you can provide more details about your specific situation, I'd be happy to give more targeted advice."""

# Initialize RAG system
rag_system = AgricultureRAGSystem()

@router.post("/query", response_model=ChatResponse)
async def advanced_chatbot_query(query: Query):
    """
    Advanced agricultural chatbot with RAG capabilities
    """
    try:
        start_time = datetime.now()
        
        # Retrieve relevant documents
        retrieved_docs = rag_system.retrieve_relevant_documents(query.question, top_k=3)
        
        if not retrieved_docs:
            # Fallback response
            return ChatResponse(
                answer="I don't have specific information about your question in my knowledge base. Could you provide more details or rephrase your question?",
                confidence_score=0.3,
                sources=[],
                related_topics=["general_farming", "crop_management"],
                response_type="fallback"
            )
        
        # Extract documents and scores
        docs, scores = zip(*retrieved_docs)
        avg_confidence = float(np.mean(scores))
        
        # Generate response using LLM with retrieved context
        llm_response = rag_system.generate_llm_response(query.question, list(docs))
        
        # Determine response type based on confidence
        response_type = "rag" if avg_confidence > 0.7 else "hybrid"
        
        # Generate related topics
        related_topics = list(set([doc.category for doc in docs]))
        
        # Store conversation in database (for analytics)
        try:
            conn = sqlite3.connect(rag_system.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (question, answer, confidence) VALUES (?, ?, ?)",
                (query.question, llm_response["answer"], avg_confidence)
            )
            conn.commit()
            conn.close()
        except Exception as db_error:
            logger.warning(f"Failed to store conversation: {db_error}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Query processed in {processing_time:.2f}s with confidence {avg_confidence:.2f}")
        
        return ChatResponse(
            answer=llm_response["answer"],
            confidence_score=avg_confidence,
            sources=llm_response["sources"],
            related_topics=related_topics,
            response_type=response_type
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing your question")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_system": "initialized",
        "knowledge_base_size": len(rag_system.knowledge_base),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/topics")
async def get_available_topics():
    """Get available topics in the knowledge base"""
    topics = list(set([doc.category for doc in rag_system.knowledge_base]))
    return {
        "available_topics": topics,
        "total_documents": len(rag_system.knowledge_base)
    }