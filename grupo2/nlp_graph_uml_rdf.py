import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from rdflib import Graph, Namespace
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import spacy
from spacy.matcher import Matcher
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler


pd.set_option("display.max_colwidth", 200)
# %matplotlib inline

candidate_sent = pd.read_csv("wikiwiki.csv", encoding="unicode_escape")
candidate_sent.shape

nlp = spacy.load("es_core_news_md")

ruler = EntityRuler(nlp)
patterns = [{"label": "ORG", "pattern": "Hoy duermo afuera"}]
ruler.add_patterns(patterns)
nlp.add_pipe(ruler)

matcher = Matcher(nlp.vocab)
nucleo = [
    {"DEP": "nsubj"}
]  # Caso 1 -"Los kayakistas contratan travesías en kayak."
nucleoMD = [
    {"DEP": "nsubj"},
    {"DEP": "amod"},
]  # Caso 2 - "Los kayakistas expertos contratan travesías en kayak."
nucleoMI = [
    {"DEP": "nsubj"},
    {"POS": "ADP"},
    {"POS": "NOUN"},
]  # Caso 3.1 - "Los kayakistas de Córdoba contratan travesías en kayak."
nucleoMIv1 = [
    {"DEP": "nsubj"},
    {"POS": "ADP"},
    {"POS": "DET"},
    {"POS": "NOUN"},
    {"POS": "ADJ"},
]
nucleoMIv2 = [
    {"DEP": "nsubj"},
    {"POS": "ADP"},
    {"POS": "DET"},
    {"POS": "NOUN"},
]  # Caso 3.2 - "Los kayakistas de la montaña contratan travesías en kayak."
nucleoMIv3 = [
    {"DEP": "nsubj"},
    {"POS": "ADP"},
    {"DEP": "nmod"},
]  # Caso 3.3 - "Los kayakistas del pantano contratan travesías en kayak."
nucleoMDMI = [
    {"DEP": "nsubj"},
    {"DEP": "amod"},
    {"POS": "ADP"},
    {"POS": "NOUN"},
]  # Caso 4.1 - "Los kayakistas expertos de Córdoba contratan travesías en kayak."
nucleoMDMIv2 = [
    {"DEP": "nsubj"},
    {"DEP": "amod"},
    {"POS": "ADP"},
    {"POS": "DET"},
    {"POS": "NOUN"},
]  # Caso 4.2 - "Los kayakistas expertos de la montaña contratan travesías en kayak."
nucleoMDMIv3 = [
    {"DEP": "nsubj"},
    {"DEP": "amod"},
    {"POS": "ADP"},
    {"DEP": "nmod"},
]  # Caso 4.3 - "Los kayakistas expertos del pantano contratan travesías en kayak."


# estos dos hay que revisar, son temporales quizás
casoEspecial = [
    {"DEP": "nsubj"},
    {"POS": "ADP"},
    {"DEP": "nummod"},
    {"POS": "NOUN"},
]
casoEspecial2 = [
    {"DEP": "nsubj"},
    {"POS": "ADP"},
    {"POS": "ADV"},
    {"POS": "ADP"},
    {"DEP": "nummod"},
    {"POS": "NOUN"},
]

"""od = [{"POS": "VERB"}, {"DEP":"obj"}]
odv2 = [{"POS": "VERB"}, {"DEP":"obj"}, {"POS":"ADP"}, {"DEP":"nmod"}]

od_oi = [{"DEP":"obj"}, {"POS":"ADP"}, {"DEP":"obj"}, {"DEP":"amod"}]
od_oi_v2 = [{"DEP":"obj"}, {"POS":"ADP"}, {"DEP":"nmod"}, {"POS":"ADP"}, {"DEP":"obj"}]
od_oi_v3 = [{"DEP":"obj"}, {"POS":"ADP"}, {"DEP":"nmod"}, {"POS":"ADP"}, {"DEP":"obj"}, {"DEP":"amod"}]"""

matcher.add("Nucleo", None, nucleo)
matcher.add("NucleoMD", None, nucleoMD)
matcher.add("NucleoMI", None, nucleoMI)
matcher.add("NucleoMIv1", None, nucleoMIv1)
matcher.add("NucleoMIv2", None, nucleoMIv2)
matcher.add("NucleoMIv3", None, nucleoMIv3)
matcher.add("NucleoMDMI", None, nucleoMDMI)
matcher.add("NucleoMDMIv2", None, nucleoMDMIv2)
matcher.add("NucleoMDMIv3", None, nucleoMDMIv3)

matcher.add("casoEspecial", None, casoEspecial)
matcher.add("casoEspecial2", None, casoEspecial2)

"""matcher.add("OD", None, od)
matcher.add("ODv2", None, odv2)

matcher.add("OD_OI", None, od_oi)
matcher.add("OD_OI_v2", None, od_oi_v2)
matcher.add("OD_OI_v3", None, od_oi_v3)"""


def buscarPosicionVerbo(doc):
    posVerbo = -1
    for i in range(0, len(doc)):
        if posVerbo == -1:
            if doc[i].pos_ == "VERB" or doc[i].lemma_ == "ser":
                posVerbo = i
    return posVerbo


# Procesar qué es "un costo" no resulta tarea sencilla. Por ahora definiré lo siguiente: todo lo que esté entre estos símbolos
# guiones representará "un costo", equivaldría a definirlo como entity pero eso requeriría cierto formato (inviable?).


def buscarUnCosto(doc):
    ent = ""
    pos1 = 0
    pos2 = 0
    index = 0
    for i in doc:
        if i.text == "<":
            pos1 = index
        if i.text == ">":
            pos2 = index
        index = index + 1
    for i in range(pos1 + 1, pos2):
        ent = ent + doc[i].text + " "
    return [ent, pos2 + 1]


# Yo tengo que encontrar la forma de la oración. Si es VERBO + OD. O VERBO+OD+OI. Si supongo que tienen ese formato, entonces
# puedo preguntar si, lo que está "a continuación del verbo" es el OD. Y si lo que está "a continuación del OD" es el OI. ¿Manejarse con las posiciones?
# Es absolutamente necesario, PARA ESTA IMPLEMENTACIÓN que la oración termine con punto. Sino producirá error.


def buscarObjetos(doc):
    posVerbo = -1
    od = ""
    oi = ""
    ind = 0
    posOD = 0
    ret = []

    # todo esto es para ver si hay una entidad costo, que sea el OD, y marco dónde terminaba el OD. Podría ser OI? Yo pienso que no, nunca responderá "a quien"
    ret = buscarUnCosto(doc)
    od = ret[0]
    posOD = ret[1]

    if od == "":
        posVerbo = buscarPosicionVerbo(doc)

        for i in range(posVerbo, len(doc)):
            if (
                (
                    doc[i].dep_ == "obj"
                    and (
                        doc[i].pos_ == "NOUN"
                        or doc[i].pos_ == "PROPN"
                        or doc[i].pos_ == "PRON"
                        or doc[i].pos_ == "ADJ"
                        or doc[i].pos == "NUM"
                    )
                )
                or doc[i].dep_ == "obl"
                or (
                    doc[i].dep_ == "ROOT"
                    and (
                        doc[i].pos_ == "NOUN"
                        or doc[i].pos_ == "ADJ"
                        or doc[i].pos_ == "PROPN"
                    )
                )
            ):
                if (
                    ind == 0
                ):  # el "ind" se está usando para marcar si había o no OD, si hay queda en 1
                    od = doc[i].text
                    if (
                        doc[i + 1].pos_ == "ADP"
                    ):  # evalúa si el OD tenía al lado un modificador indirecto
                        if doc[i + 2].dep_ == "nmod":
                            od = (
                                od
                                + " "
                                + doc[i + 1].text
                                + " "
                                + doc[i + 2].text
                            )
                            i = i + 2
                    else:  # o un modificador directo
                        if doc[i + 1].pos_ == "ADJ":
                            od = od + " " + doc[i + 1].text
                            i = i + 1
                    ind = ind + 1
                    posOD = i + 1

    if doc[posOD].text != ".":
        # Evaluará hallar un OI DESPUÉS del OD (para eso usa posOD)
        if (
            doc[posOD].pos_ == "ADP"
        ):  # El OI está encabezado por la preposición
            for i in range(posOD, len(doc)):
                if (
                    doc[i].dep_ == "nummod"
                    or doc[i].dep_ == "obj"
                    or doc[i].dep_ == "nmod"
                    or doc[i].dep_ == "obl"
                ):
                    oi = doc[i].text
                    if (
                        doc[i + 1].pos_ == "ADP"
                    ):  # evalúa si el OI tenía al lado un modificador indirecto
                        if doc[i + 2].dep_ == "nmod":
                            oi = (
                                oi
                                + " "
                                + doc[i + 1].text
                                + " "
                                + doc[i + 2].text
                            )

                    else:  # o un modificador directo
                        if (
                            doc[i + 1].pos_ == "ADJ"
                            or doc[i + 1].dep_ == "nmod"
                            or doc[i + 1].dep_ == "amod"
                        ):
                            oi = oi + " " + doc[i + 1].text

    return [od, oi]


def get_ent(sent):
    ent = ""
    for i in sent.ents:
        ent = ent + i.text
    return ent


def get_entities(sent):
    ent1 = ""
    ent2 = ""
    obj = []
    ent = ""

    sent = nlp(sent)
    ent = get_ent(sent)  # Esto es reemplazable por un matcher

    # Me fijo si encontré ent es porque descubrí una entidad, podría ser el núcleo? En realidad debería fijarse si es parte del sujeto esa entidad (si está antes del verbo?)
    # Hay un matcher para esto. No logré hacer que funcione, puse {"ENT_TYPE":"ORG"}
    if ent != "":
        ent1 = ent
    else:
        # Mando a matchear con todos los patrones posibles. Devolverá el sujeto que sea correcto.
        matches = matcher(sent)
        span = ""
        for _, start, end in matches:
            span = sent[start:end]  # The matched span
        ent1 = str(span)

    obj = buscarObjetos(sent)
    if (
        obj[1] != ""
    ):  # si tiene OI, la 2da ent es el OI porque la relacion va a ser el verbo+od
        ent2 = obj[1]
    else:
        ent2 = obj[0]  # sino, la 2da ent es el od como siempre

    return [ent1.strip(), ent2.strip()]


# La relación está determinada meramente por los objetos directo e indirecto. Por lo tanto, se analiza lo siguiente:
# si tiene OI, la relación va a ser VERBO+OD
# si no tiene OI, la relación va a ser el VERBO
# Observar que se analiza "ser" como excepción
def get_relation(sent):
    doc = nlp(sent)
    rel = ""

    for i in doc:
        if i.lemma_ == "ser":
            rel = i.text
            if (
                i.nbor().pos_ == "VERB"
            ):  # sentencia para analizar formas verbales compuestas. ej: "es considerado", "fue medido"
                rel = i.text + " " + i.nbor().text
        if rel == "" and i.pos_ == "VERB":
            rel = i.text

    obj = buscarObjetos(doc)

    if obj[1] != "":
        # print("tiene oi")
        rel = rel + " " + obj[0]

    return rel.strip()


entity_pairs = []

for sentence in candidate_sent["sentence"]:
    entity_pairs.append(get_entities(sentence))

relations = [get_relation(i) for i in candidate_sent["sentence"]]

# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

lemma = [nlp(i[0])[0].lemma_ for i in entity_pairs]

entity_pairs[0:30]

df = pd.DataFrame(
    {"Entidad1": source, "relacion": relations, "Entidad2": target}
)
print(df)
plt.figure(figsize=(12, 12))
G = nx.from_pandas_edgelist(
    df=df,
    source="Entidad1",
    target="Entidad2",
    edge_attr="relacion",
    create_using=nx.DiGraph(),
)
pos = nx.spring_layout(
    G, k=5
)  # nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
nx.draw(G, pos, with_labels=True, node_color="pink", node_size=2000)
labels = {e: G.edges[e]["relacion"] for e in G.edges}
nx.draw_networkx_edge_labels(
    G, pos, edge_cmap=plt.cm.Blues, edge_labels=labels
)
plt.show()

# esto es para saber si la relacion es verbo +od o verbo solo
def determinarRelacion(rel, ent2):
    relacion = nlp(rel)

    temp = ""

    tieneOI = False
    for i in relacion:
        if i.dep_ == "obj" or i.dep_ == "nsubj":
            tieneOI = True

        if i.dep_ == "ROOT":
            temp = temp + " " + str(i.lemma_).strip()
        else:
            temp = temp + " " + str(i).strip()

    if not tieneOI:
        rel = temp + " " + ent2
    else:
        rel = temp
    return rel.strip()


class Clase:
    def __init__(self):
        self.nombre = ""
        self.variablesInstancia = []
        self.metodos = []
        self.subclases = []


def clase_existente(lista_clases, nombre_clase):
    for clase in lista_clases:
        if nombre_clase == clase.nombre:
            return clase
    return None


def es_variable(relacion):
    rel = nlp(relacion)
    return (rel[0].lemma_ == "tener") and (len(rel) == 1)


def buscar_subclases(lemma):
    # Despues las subclases
    subclases = []
    for clase in lista_clases:
        if not clase.nombre in dicLemma.values():
            if nlp(clase.nombre)[0].lemma_ == lemma:
                subclases.append(clase)
    return subclases


def es_subclase(buscada, lista_final):
    for clase in lista_final:
        if buscada in clase.subclases:
            return True
    return False


# PROCESAMIENTO 1
# ------------------------------------------------------------------------------------------------------------------
# A partir del df, hago una lista de posibles clases, con sus respectivos métodos y variables


lista_clases = []
for index, row in df.iterrows():
    existe = clase_existente(lista_clases, row["Entidad1"])
    if existe == None:
        nueva = Clase()
        nueva.nombre = row["Entidad1"]
        if es_variable(row["relacion"]):
            nueva.variablesInstancia.append(row["Entidad2"])
        else:
            rel = determinarRelacion(row["relacion"], row["Entidad2"])
            nueva.metodos.append(rel)
        lista_clases.append(nueva)
    else:
        if es_variable(row["relacion"]):
            existe.variablesInstancia.append(row["Entidad2"])
        else:
            rel = determinarRelacion(row["relacion"], row["Entidad2"])
            existe.metodos.append(rel)


# PROCESAMIENTO 2
# ---------------------------------------------------------------------------------------------------------------------------
# Establezco jerarquias.. Por cada una de las enntidades, voy a mirar si el ENTIDAD=LEMA.
# Una vez que encuentro ENTIDAD=LEMA, voy a recorrer la lista buscando todas las jerarquias y reacomodando los ptros

dicLemma = dict()
for clase in lista_clases:
    dicLemma[clase.nombre] = nlp(clase.nombre)[0].lemma_

lista_final = (
    []
)  # Mi idea inicial era usar el método remove pero rompí todo. Abierto absolutamente a quien quiera intentarlo. Tiene que remover las "subclases" que encuentra de la lista_clases, luego de hacer que la clase las agregue como sus subclases

for clase in lista_clases:
    if (
        dicLemma[clase.nombre] == clase.nombre
    ):  # procesara las de mismo lema y nombre
        clase.subclases = buscar_subclases(dicLemma[clase.nombre])
        if len(clase.subclases) > 0:
            lista_final.append(clase)

# Una vez que establecí jerarquías, proceso todo lo que no sea jerarquía (ni clase, ni subclase de esa clase)
for clase in lista_clases:
    if not es_subclase(clase, lista_final) and not clase in lista_final:
        cantidad_variables = len(clase.variablesInstancia)
        cantidad_metodos = len(clase.metodos)
        if (cantidad_variables + cantidad_metodos) > 1:
            lista_final.append(clase)

# Set con todos los métodos para eliminar repeticiones
for clase in lista_final:
    for subclase in clase.subclases:
        subclase.metodos = set(subclase.metodos)
    clase.metodos = set(clase.metodos)

data = []
dataSource = []
for i in relations:
    doc = nlp(i)
    for k in doc:
        if k.pos_ == "NUM":
            data.append(i)
            index = relations.index(i)
            dataSource.append(source[index])
for clase in lista_final:
    print("CLASE:", clase.nombre)
    source.append(clase.nombre)
    target.append("Class")
    relations.append("Is_a")
    print(" - Variables de instancia: ", end="")
    for variable in clase.variablesInstancia:
        print(variable, end=", ")
        source.append(variable)
        target.append("instance_var_" + clase.nombre)
        relations.append("Is_a")
    print()
    print("  - Metodos: ", end="")
    for metodo in clase.metodos:
        print(metodo, end=", ")
        source.append(metodo)
        target.append("method_of_" + clase.nombre)
        relations.append("Is_a")
    print()

    for subclase in clase.subclases:
        print(" SUBCLASE:", subclase.nombre)
        source.append(subclase.nombre)
        target.append("subclass_of_" + clase.nombre)
        relations.append("Is_a")
        print(" - Variables de instancia: ", end="")
        for variable in subclase.variablesInstancia:
            print(variable, end=", ")
            source.append(variable)
            target.append("instance_var_" + subclase.nombre)
            relations.append("Is_a")
        print()
        print("  - Metodos: ", end="")
        for metodo in subclase.metodos:
            print(metodo, end=", ")
            source.append(metodo)
            target.append("method_of_" + subclase.nombre)
            relations.append("Is_a")
        print()
    print()
    print(
        "-----------------------------------------------------------------------------"
    )
for i in data:
    source.append(i)
    relations.append("MethodElement")
    target.append(dataSource[data.index(i)])
# class---> entity 1: ej:travesias relation: es una entity 2: clase
# subclass---> entity 2: kayakista inexperto relation: es una entity2:subclass_kayakista
# methods---> entity 1:solicitar_cotizacion() relation : es un entity:2 method_travesia
# instance_var----> entity1:duracion relation:es una entity2: instance_var_travesias_en_kayak
# -------------------------------------template-----------------------------------------------
# class------> entity1: ENT1 relation: Es una      ENT2: clase
# subclass --------> entity1: EN1 relation: Es una ENT2: subclase_ENT2
# methods ---------> ENT1: EN1 relation: Es un ENT:metodo_ENT2
# instance_var --------> Ent1:EN1 relation: Es una ENT2: variable_de_instancia_ENT2
# --------------------------------------------------------------------------------------------
# methodbody, methodElement to do list
# entity1: 100 pesos relation: methodElement entity2: costo
df = pd.DataFrame(
    {"Entidad1": source, "relacion": relations, "Entidad2": target}
)
print(df)
plt.figure(figsize=(12, 12))
G = nx.from_pandas_edgelist(
    df=df,
    source="Entidad1",
    target="Entidad2",
    edge_attr="relacion",
    create_using=nx.DiGraph(),
)
pos = nx.spring_layout(
    G, k=5
)  # nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
nx.draw(G, pos, with_labels=True, node_color="pink", node_size=2000)
labels = {e: G.edges[e]["relacion"] for e in G.edges}
nx.draw_networkx_edge_labels(
    G, pos, edge_cmap=plt.cm.Blues, edge_labels=labels
)
plt.show()


file = open("Test.txt", "w")
##print("\"")

file.write("@startuml" + os.linesep)  # inicio del archivo :D kk
for clase in lista_final:
    file.write('class "' + clase.nombre + '" {' + os.linesep)
    for variable in clase.variablesInstancia:
        file.write("Variable de instancia : " + variable + os.linesep)
    for metodo in clase.metodos:
        kk = ""
        doc = nlp(metodo)
        for i in doc:
            kk = kk + i.text + "_"
        file.write(kk + "()" + os.linesep)
    file.write("}" + os.linesep)  # cierro clase
    for subclase in clase.subclases:
        kk = ""
        doc = nlp(subclase.nombre)
        for i in doc:
            kk = kk + i.text + "_"
        file.write(
            '"'
            + clase.nombre
            + '" <|-- '
            + '"'
            + subclase.nombre
            + '"'
            + os.linesep
        )
        file.write('class "' + subclase.nombre + '" {' + os.linesep)
        for variable in subclase.variablesInstancia:
            file.write("Variable de instancia : " + variable + os.linesep)
        for metodo in subclase.metodos:
            kk = ""
            doc = nlp(metodo)
            for i in doc:
                kk = kk + i.text + "_"
            file.write(kk + "()" + os.linesep)
        file.write("}" + os.linesep)
file.write("@enduml" + os.linesep)
file.close()

os.system("java -jar plantuml.jar test.txt")


# (subject0, predicate0, object0)
# (entidad1,relacion,entidad2) para nosotros
# create a Graph
gf = Graph()
EX = Namespace("http://example.org/")
# print(len(source))
for i in range(0, len(source)):
    # print((source[i],relations[i],target[i]))
    kk = ""
    doc = source[i].split()
    for j in doc:
        kk = kk + j + "_"
    gg = ""
    doc = target[i].split()
    for j in doc:
        gg = gg + j + "_"
    zz = ""
    doc = relations[i].split()
    for j in doc:
        zz = zz + j + "_"
    gf.add((EX[kk], EX[zz], EX[gg]))
    # g.add(Literal(relations[i]))
    # g.add(Literal(target[i]))
# RDF.type
bob = EX["bob"]
alice = EX["alice"]
# gf.add((bob,RDF.type,FOAF.Person))
# gf.add((bob,RDF.type,alice))
print(gf.serialize(format="n3").decode("utf-8"))


G = rdflib_to_networkx_multidigraph(gf)

# Plot Networkx instance of RDF Graph
pos = nx.spring_layout(G, scale=1900)
edge_labels = nx.get_edge_attributes(G, "r")
# nx.draw_networkx_edge_labels(G, pos, labels=edge_labels)
# nx.draw(G, with_labels=True)
pos = nx.spring_layout(
    G, k=5
)  # nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
nx.draw(
    G,
    with_labels=True,
    node_color="skyblue",
    node_size=1500,
    edge_cmap=plt.cm.Blues,
    pos=pos,
)
plt.show()

print(data)
print("---")
print(dataSource)
