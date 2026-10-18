from langchain_text_splitters import(
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

text="""
Those with no cultivation talent are bound to stay as mortals, the lowest level in the society, who could be trampled upon and toyed with but those who have the talent to cultivate can become Gu Masters, going beyond the ordinary. A Gu Master is someone who has opened their mortal aperture within their body, can use primeval essence to manipulate Gu and use their wondrous abilities.

Gu Masters cultivate their chosen path, research, innovate, and create new Gu which makes them stronger, better and more adapt for future cultivation and their pursuit to understand the Great Dao, through all those acts of using Gu. Not only (variant-) humans but also beasts can open their aperture and cultivate. Beast Gu Masters that have the wisdom of a human are most often superior compared to human Gu Masters of the same rank, due to their more resilient bodies, high vitality, regenerative abilities, better vision and hearing, greater physical prowess, and longer lifespan. 

The aperture is a mysterious and enigmatic spherical space located inside the Gu Master’s body around the stomach's area. Due to it being ethereal it does not occupy physical space such as an internal organ would. The aperture is endlessly huge, yet at the same time infinitely small. Some call it the Purple Prefecture, some call it the Chinese Pool, Essence Pools, or Purple Residences, however, many know it as the Primeval Sea Aperture. Before becoming a Gu Master the mortal has to awaken their aperture, aside from some special methods, this is mostly done so by coming into contact with Hope Gu.


The aperture has three specialized uses for its Gu Master which are:

    Producing and storing primeval essence depending on the Gu Master’s aptitude and rank.
    Serving as the foundation of a Gu Master's cultivation and the basis to promote their cultivation with the help of primeval essence.
    Serving as a place for a Gu Master to put and safely store their Gu.
"""

#fixed sized chiunk

fixed = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0
)

print("fixed size")

for i, chunk in enumerate(fixed.split_text(text), 1):
    print(f"\nchunk {i}:")
    print(chunk)
    print("=========================")


paragraph=CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0
)

print("paragrarg")

for i,chunk in enumerate(paragraph.split_text(text), 1):
    print(f"\nchunk {i}:")
    print(chunk)
    print("===============")

print("recursice")

recursive=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

for i, chunk in enumerate(recursive.split_text(text), 1):
    print("\n chunk{i}:")
    print(chunk)
