QMARK = '?'
NULL = '∅'


# Determines whether h1 is more general than h2
def more_general(h1, h2):
    is_more_general = []  # Boolean values for each feature where True = more general and False = less general
    for f1, f2 in zip(h1, h2):
        is_more_general.append(
            f1 == QMARK or (f1 != NULL and (f1 == f2 or f2 == NULL))
        )

    return all(is_more_general)


def match(d, h):
    return more_general(h, d)


def minimal_generalisations(h, x):
    h_new = list(h)
    for i in range(len(h)):
        if not match(x[i:i+1], h[i:i+1]):  # Check that h matches d
            h_new[i] = QMARK if h[i] != NULL else x[i]
    return [tuple(h_new)]

# def minimal_generalisations(h1, h2):
#     h = h1
#     for i in range(len(h)):
#         if not match(h2[i], h1[i]):  # if the elements of h1 and h2 dont match
#             # if h1[i] == '∅' set h[i] to h2[i], if h1[i] and h2[i] are not equal set h[i] to '?'
#             h[i] = QMARK if h1[i] != NULL else h2[i]
#     return h


def minimal_specialisation(h, D, x):
    results = []
    for i in range(len(h)):
        if h[i] == QMARK:
            for feature in D[i]:
                if x[i] != feature:
                    h_new = h[:i] + (feature,) + h[i + 1:]
                    results.append(h_new)
        elif h[i] != NULL:
            h_new = h[:i] + (NULL,) + h[i + 1:]
            results.append(h_new)
    return results


def cea_trace(domains, training_examples):
    S = {tuple(NULL for _ in range(len(domains)))}  # Most-Specific hypothesis in H
    G = {tuple(QMARK for _ in range(len(domains)))}  # Most-General hypothesis in H
    S_trace = [S]
    G_trace = [G]

    for d in training_examples:  # for each training example, d, do
        G = G.copy()
        S = S.copy()
        dx, dy = d
        if dy:  # if d is positive
            G = {g for g in G.copy() if match(dx, g)}  # remove from G any hypothesis that do not match d
            for s in S.copy():  # for each hypothesis s in S that does not match d
                if not match(dx, s):
                    S.remove(s)  # Remove s from S
                    for h in minimal_generalisations(s, dx):  # Add to S all minimal generalisations, h, of s such that:
                        # 1) h matches d
                        # 2) some member of G is more general than h
                        if any([more_general(g, h) for g in G.copy()]):  # fix
                            S.add(h)
            # remove from S any h that is more general than another hypothesis in S
            S = {sj for si in S.copy() for sj in S.copy() if more_general(si, sj)}  # fix # only allow hypotheses of equal generality
        elif not dy:  # if d is negative
            S = {s for s in S.copy() if not match(dx, s)}  # remove from S any hypothesis that match d
            for g in G.copy():  # for each hypothesis g in G that matches d
                if match(dx, g):
                    G.remove(g)  # Remove g from G
                    for h in minimal_specialisation(g, domains, dx):  # Add to G all minimal specialisations, h, of g such that:
                        # 1) h does not match d
                        # 2) some member of S is more specific than h
                        a = any([more_general(h, s) for s in S])
                        if a:
                            G.add(h)
            # remove from G any h that is more specific than another hypothesis in G
            G = {gj for gi in G.copy() for gj in G.copy() if more_general(gj, gi)}  # only allow hypotheses of equal generality

        # Update the traces
        S_trace.append(S.copy())
        G_trace.append(G.copy())
    return S_trace, G_trace

def main():
    # # Example 1
    # domains = [
    #     {'red', 'blue'}
    # ]
    #
    # training_examples = [
    #     (('red',), True)
    # ]
    #
    # S_trace, G_trace = cea_trace(domains, training_examples)
    # print(len(S_trace), len(G_trace))
    # print(all(type(x) is set for x in S_trace + G_trace))
    # S, G = S_trace[-1], G_trace[-1]
    # print(len(S), len(G))

    # Expect
    # 2 2
    # True
    # 1 1

    # # Example 2
    # domains = [
    #     {'T', 'F'}
    # ]
    #
    # training_examples = []  # no training examples
    #
    # S_trace, G_trace = cea_trace(domains, training_examples)
    # print(len(S_trace), len(G_trace))
    # S, G = S_trace[-1], G_trace[-1]
    # print(len(S), len(G))

    # Example 3
    # domains = [
    #     ('T', 'F'),
    #     ('T', 'F'),
    # ]
    #
    # training_examples = [
    #     (('F', 'F'), True),
    #     (('T', 'T'), False),
    # ]
    #
    # S_trace, G_trace = cea_trace(domains, training_examples)
    # print(len(S_trace), len(G_trace))
    # S, G = S_trace[-1], G_trace[-1]
    # print(len(S), len(G))



    # matches = more_general(('Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'), (NULL, NULL, NULL, NULL, NULL, NULL))
    # print(matches)

    # # Example 4
    # domains = [
    #     ('Sunny', 'Cloudy', 'Rainy'),  # Sky
    #     ('Warm', 'Cold'),  # Temp
    #     ('Normal', 'High'),  # Humidity
    #     ('Strong', 'Weak'),  # Wind
    #     ('Warm', 'Cool'),  # Water
    #     ('Same', 'Change')  # Forecast
    # ]
    #
    # training_examples = [
    #     (('Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'), True),
    #     (('Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'), True),
    #     (('Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'), False),
    #     (('Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'), True),
    # ]
    #
    # S_trace, G_trace = cea_trace(domains, training_examples)

    # Example 5
    domains = [
        {'red', 'green', 'blue'}
    ]

    training_examples = [
        (('red',), True),
        (('green',), True),
        (('blue',), False),
    ]

    S_trace, G_trace = cea_trace(domains, training_examples)
    S, G = S_trace[-1], G_trace[-1]
    print(len(S) == len(G) == 0)


if __name__ == "__main__":
    main()