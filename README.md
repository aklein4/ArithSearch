# ArithSearch

### TODO:

1. Port sample_search and its wrappers, recursive_search and splitting_search, from c_star.py to C++.
2. Make it DivSearch return visualizable solutions.
3. Integrate stochastic methods (ex. sample_search) with DivSearch.
4. Port DivSearch to C++.

### MetaSearch Example:

1b^2.0c^2.0 + 1ac + 1ab + 1ab^3.0c^3.0 + 1abc^2.0 + 1a + 1abc + 1a^3.0c^4.0 + 1ab^3.0c

For the polynomial:

 1a^2c^3 + 1a^5b^2c^12 + 1a^6b^3c^6 + 1a^10c + 1a^11 + 1a^4bc^11 + 1a^10b^3c^12 + 1a^5bc^10 + 1ab^11c^13 + 1a^6b^4

The best cost that regular Horner's Scheme can find is 87 with an order of (c, a, b) - note that for some reason some orders didn't run, but this can be redone:

 --- Regular --- <br />
(a, c, b) --> 93<br />
(b, a, c) --> 117<br />
(c, a, b) --> 87<br />
(c, b, a) --> 96

For MetaSearch, greedySearch with default parameters was able to find a solution with cost 114. Using annealSearch (5000 iterations, gamma=0.25, temp_start=10, temp_schedule=4000), a cost 76 solution was found. Using randomSearch (1000 iterations, gamma=0.25), a cost 84 solution was found.

![annealSearch output graph](./anneal_example.png)

![randomSearch output graph](./random_example.png)

Note that for all of the listed methods, examples can found for which that method does the best. Also note that caching methods were not used anywhere in this test for a fair comparison, and that if they were used then MetaSearch would likely have much lower costs.

### DivSearch Proof of Superiority:

For the polynomial:

3a + 5a^3bc^3 + 7b^2c + 11a^3c^4 + 13ab^2c + 17abc^2 + 19a^2c^2 + 23a^5b^2c^2 + 29b^5c^4 + 31ab^5c

The lowest possible number of multiplications using regular Horner's scheme, basic_horner (without intermediate variables), is 34, with the order of (c, b, a):

 --- Regular --- <br />
(a, b, c) --> 38<br />
(a, c, b) --> 35<br />
(b, a, c) --> 37<br />
(b, c, a) --> 35<br />
(c, a, b) --> 33<br />
(c, b, a) --> 34

Using the improved Horner's scheme, DivSearch (without intermediate variables), a solution of cost 30 was found:

 --- Improved --- <br />
None --> 30

This proves that polynomials exist such that:
1. DivSearch's solution-space is a STRICT superset of basic_horner's solution-space (DivSearch's solution-space being a superset of basic_horner's solution-space can be proven by inspection).
2. The minimum multiplication cost of DivSearch's solution-space is strictly lower than the minimum cost of basic_horner's solutions-space (the opposite cannot be true since DivSearch's solution-space is a superset of basic_horner's solution-space).

When mutliplication by scalars is allowed to be broken into a sequence of additions, examples can be found such that these facts still hold.

### Example:

The DivSearch function in DivSearch.py was able to solve the following polynomial using 2815 multiplications, while the benchmarking library requires 10061 multiplications and a naive representation would require about 47900.

1a^28b^26c^19d^35e^38f^34g^29h^16i^5j^35k^5l^20m^39n^37o^33 + 1a^4b^13c^33d^16e^22f^9g^15h^32i^25j^6k^24l^14m^29n^11o^36 + 1a^30b^27c^6de^12f^39g^22i^38j^13k^10l^31m^22n^12o^17 + 1a^10b^16c^26d^9e^24f^17g^26h^21i^10j^10k^5l^17m^30n^24o^29 + 1a^16b^30c^35d^17e^35f^21g^17h^6i^21j^11k^33l^11m^5n^16o^8 + 1a^12b^29c^35d^18e^15f^24g^12h^11i^24j^39k^26l^15m^8n^29o^26 + 1a^25b^37c^24d^35e^13f^27g^14h^31i^9j^8k^38l^38m^3n^8o^5 + 1a^26b^18c^27d^15e^27f^14g^29h^28i^7j^13k^19l^12m^36n^38o^25 + 1a^2b^6c^26d^7e^12f^17g^7h^6i^17j^18k^10l^13m^29n^14o^16 + 1a^22b^6cd^10e^34f^31g^31h^30i^23j^15k^38l^37m^14n^39o^28 + 1a^29b^31d^26e^37g^24h^39i^20j^30k^18l^2m^5n^35o^9 + 1a^31b^6c^12d^38e^39f^6g^9h^3i^39j^26k^7l^37m^19n^28o^5 + 1a^24b^35c^28d^5ef^26g^14h^13i^12j^29k^33l^30m^16n^9o^35 + 1a^21b^7c^23d^33e^22f^4g^17h^35i^21j^2k^12l^29m^34n^15o^22 + 1a^36b^26c^3d^7e^27f^21g^18h^11i^15j^26k^32l^31m^10n^17o^37 + 1a^22b^15c^8d^8e^35f^39gh^6i^38j^3k^2l^32m^13n^8o^20 + 1a^13b^23c^13de^35f^39g^16h^19i^37j^9k^39l^33m^19n^26o^29 + 1a^33b^29c^37d^11e^32f^18g^14h^19i^24j^7k^23l^8m^5 + 1a^25b^36c^9d^39e^32f^36g^26h^30i^18j^2k^17l^22m^36n^24o^11 + 1a^18b^23c^13d^26e^14g^27h^22i^4j^5k^28l^9m^3n^6o^38 + 1a^34b^8c^15d^34e^21f^34g^12h^7i^31j^17k^12l^28m^27n^7o^8 + 1a^2b^25c^11d^2e^13f^15g^8h^9i^3j^14k^37l^38m^39n^9o^12 
+ 1a^2b^20c^32d^17e^35f^7g^12h^26i^3j^29k^39l^35m^3n^33o^9 + 1a^28b^19c^32d^28e^18f^31g^6h^27i^18j^13k^24lm^29n^2o^18 + 1a^30b^38c^8d^25e^24f^21g^27h^9i^19j^6k^22l^15m^17n^15o^34 + 1a^30b^38c^39d^11e^39f^19g^16h^14i^14j^30k^8l^12m^25n^29o^33 + 1a^36b^15c^30d^13e^33f^12g^32h^38i^33j^17k^30l^36m^30n^7o^19 + 1a^17b^15c^35d^19e^16f^19g^10h^33i^15j^27k^6l^6m^7n^8o^28 + 1a^19b^15c^30d^8e^4f^6g^31h^36i^15j^23k^12l^36m^26n^27o^18 + 1a^28b^6c^33d^7e^15f^34g^7h^7i^17j^6k^25l^38m^27n^30o^17 + 1a^15b^16c^29e^31f^11g^32i^35j^21k^14l^13m^32n^17o^10 + 1a^6b^8c^17d^36e^33f^6g^12h^16i^17j^22k^11l^39m^23n^33o + 1a^37b^27c^8d^4e^8f^24g^36h^2i^17j^33k^10l^21m^37n^8o^22 + 1a^31b^29c^7d^37e^11f^36g^32h^11i^23j^5k^13l^3m^2n^2o^30 + 1a^17b^20c^38d^30e^38fg^12h^12i^8j^19k^5l^12m^21n^2o^28 + 1a^21b^17c^8d^34e^30fg^26h^34i^4j^13k^15l^34m^30n^29o^17 + 1a^18b^17c^34d^8e^10f^19g^22h^12i^35j^6k^21l^32m^5n^14o^17 + 1a^25b^13c^4d^27e^15f^36g^21h^9i^22j^25m^30n^33o^18 + 1a^29b^35c^39d^19e^30f^14g^16h^22i^24j^24k^30l^31m^14n^17o^18 + 1a^12b^39c^36d^11e^3f^2g^7h^7i^36j^25k^24lm^24n^14o^27 + 1a^7b^28c^13d^22e^4f^6g^24h^39i^9j^27k^30l^20m^33n^39o^34 + 1a^29b^23c^20d^33e^2g^37h^11i^33jk^6l^34m^13n^36o^34 + 1a^33b^20c^20d^4e^6f^23g^30h^23i^14jk^33l^2m^7no^24 + 1a^28b^30c^34d^9e^5f^23gh^35i^18j^35k^18l^17m^9n^18o^3 + 1a^6b^6c^37d^2e^38f^14g^32h^28i^15j^16k^33l^38m^38n^5o^34 + 1a^22b^14c^18d^16e^36f^14g^11h^33i^11j^29k^4l^6m^8n^36o^17 + 1a^12b^3c^27d^11e^8f^32g^9h^12i^8j^14k^36l^29m^31n^7o^4 + 1a^13b^29c^7de^6f^39g^33h^13i^29j^38k^28l^17m^10n^8o^27 + 1a^12b^24c^37d^18e^11f^39g^37h^34i^33j^29k^34l^26m^37n^3o^17 + 1ab^5c^17d^39e^13f^36g^31h^7i^39j^14k^30l^2m^38n^15 + 1a^11b^23c^12d^4e^32f^32g^5h^20i^37j^30k^10l^26m^22n^26o^23 + 1a^16b^38c^4d^3e^7f^30g^34h^38j^29k^33l^8m^8no^37 + 1a^4b^4c^31d^14e^20fg^34h^3i^21j^11k^26l^22m^9n^29o^17 + 1a^15b^23cd^6e^37f^14g^36h^6i^26j^37k^23l^15m^35n^23o^32 + 1a^34b^35c^19d^6e^7f^30g^20h^4i^4j^8k^32l^10m^20n^19o^2 + 1a^2b^18c^34d^33e^34f^34g^13h^32i^19jk^16l^2m^38n^25o^2 + 1b^16c^23d^8e^25f^6g^33h^5i^31j^2k^11l^17m^31n^4o^16 + 1a^9b^36c^32d^36e^7f^37g^4h^27i^18j^37k^7l^29m^36n^8o^20 + 1a^9b^5c^27d^11ef^7g^7h^13i^32j^2k^4l^9m^12n^35o^12 + 1a^11b^28c^14d^7e^29f^28g^39h^38i^16j^26k^13l^9m^18n^12o^6 + 1a^29b^15c^9d^15e^11f^2h^15i^19j^14k^16l^28m^25n^33 + 1a^11b^34c^23d^4e^24f^28g^30h^6i^39j^23k^19l^5m^31n^5o^14 + 1a^14b^18c^20d^39e^34f^38g^36h^2i^2j^38k^36l^27m^19n^8o^17 + 1a^16b^23c^15d^5e^11f^13g^24h^12i^12j^25kl^16m^31n^15o^9 + 1a^28b^26c^4d^5e^29f^2g^27h^19i^39j^6k^37l^33m^5n^29o^24 + 1a^11b^24c^32d^26e^16f^19g^21h^28i^18jk^25l^24m^12n^18o^27 + 1a^29bc^36d^27ef^9g^28h^38i^26j^38k^33l^9m^23n^31o^37 + 1a^18b^7c^13d^23e^39f^22g^35h^14i^8j^10k^34l^12m^33n^20o^30 + 1a^18b^3c^14d^6e^8f^23g^39h^25i^35j^23k^31l^38m^12n^19o^30 + 1a^33b^11c^16d^14e^32f^16g^26h^12i^10j^29k^17l^35m^38n^37o^15 + 1a^22b^38c^13d^19e^2f^6g^20h^14i^38j^35k^15l^2m^19n^36o^23 + 1a^3b^9c^19d^14e^24f^38g^15h^19i^33j^30k^8l^7m^36n^32o^37 + 1a^4b^12c^16d^2e^24f^9g^21h^37i^22j^7k^12l^18m^20n^5o^35 + 1a^22b^14c^29d^27e^14f^14g^28h^21i^31j^37k^30l^18m^33n^18o^4 + 1a^4b^9c^15d^2e^11f^18g^38h^30i^27j^16k^3l^33m^2n^15o^31 + 1a^3b^14c^2d^13e^7f^6g^5h^22i^10j^33k^22l^21m^16n^38o^10 + 1a^27b^37c^25d^27e^5f^16g^29h^10i^4k^11l^37m^3n^14o^12 + 1a^35b^24c^20d^24e^16f^5g^19h^29i^31j^4k^5l^22m^29n^25o^22 + 1a^31b^11c^24d^24e^10f^26g^7h^33i^24j^31k^30l^5m^15n^3o^35 + 1a^2b^37c^6d^33e^38f^4g^31h^11i^11j^15k^7l^33m^8n^6o^4 + 1a^22b^22c^22d^23e^21f^10g^4h^3i^4j^6k^37l^39m^3n^9o^16 + 1a^28b^29c^23d^5e^25f^3g^3h^18i^15j^21k^5l^21m^3n^22o^19 + 1a^2b^30c^4d^31e^4f^18g^10h^35i^9j^22k^6l^3m^3n^28o^4 + 1a^31b^38c^24d^37e^39f^8g^39h^29i^23k^17l^3m^4n^15o^4 + 1a^31b^17c^36d^24e^22f^17g^35h^33i^21j^38k^22l^9m^12n^31o^27 + 1a^23b^2c^6d^38e^37f^34g^15h^33i^10j^26k^31l^12o^8 + 1a^28b^12c^35d^10e^9f^10g^21h^33i^17j^2k^7l^28m^12n^7o^11 + 1a^31b^24c^9d^38e^8f^22g^18h^20i^10j^17k^19l^28m^30n^38o^23 + 1a^29b^18c^22d^19e^32f^21g^31h^6i^39j^18k^35l^23mn^17o^9 + 1a^32b^14c^32d^12e^8f^18g^29h^38i^28j^32k^21l^15m^15n^37o^28 + 1a^19c^15d^38e^14f^4gh^18i^27j^36k^37l^3m^8n^38o^24 + 1a^10b^21c^12d^39e^37f^25g^20h^38i^24j^5k^3m^9n^4o^22 + 1b^14c^38d^25f^22g^27h^7i^5j^34k^7l^18m^33n^17o^26 + 1a^9b^16c^38d^12e^33f^17g^28h^24i^19j^12k^3l^11m^32n^9o^34 + 1a^26b^5c^2d^21e^16f^10g^16h^4i^35j^14k^29l^24m^18n^30o^29 + 1a^3b^36c^6d^12e^9f^36g^9h^23i^38j^15k^3l^21m^4n^23o^34 + 1a^5b^26c^29d^13e^8f^8g^22h^28i^7j^34k^34lm^6n^18o^38 + 1a^23b^25c^20d^36e^38f^35g^24h^27i^19j^18k^32l^25m^2n^10o^19 + 1a^19b^17c^14d^18e^20f^29g^21h^27i^35k^5l^10m^20n^26o^12 + 1a^27b^2c^27d^33e^30f^8g^17h^34i^28j^39k^26l^3m^25n^32o^2 + 1a^16b^20c^25d^6e^30f^12g^25i^27j^13k^9l^27m^13n^11o^19 + 1a^33bc^26d^28e^37f^15g^7h^37i^10j^5k^5l^5m^11n^28o^24 + 1ab^25c^7d^4e^35f^19g^29h^25i^30j^13k^31m^32no^15 + 1a^12b^30c^17d^35e^26f^35g^6h^21i^4j^3k^28l^37m^11n^38o^22 + 1a^27bcde^18f^36g^5hi^33k^28l^9m^17n^7o^33 + 1a^39b^39c^22d^38e^22fg^6h^14i^8j^20k^16l^19m^20n^24o^7 + 1a^24b^16c^39d^24e^19f^30g^17h^37i^19j^24k^28l^39m^38o^7 + 1ab^36c^21d^19e^22f^9g^38h^25i^12j^20k^38l^9m^24n^9o^14 + 1a^8b^14c^24d^27e^5f^38g^36h^34i^16k^35l^14m^3n^37o^22 + 1a^20b^2c^10d^13e^33f^8g^33h^9i^36j^17k^11l^35m^7n^7o^4 + 1a^4b^15c^32d^28e^19f^7h^29i^12j^8k^23l^5m^36n^8o^23 + 1a^3b^28c^24de^39f^19g^29h^20i^10j^14k^11l^14m^9n^15o^15 + 1a^6b^20c^38d^9e^27f^13g^19h^18i^2j^17k^38l^6m^21n^23o + 1a^19b^38cd^31e^18f^9g^23h^3i^15j^23k^2l^13n^16o^24 + 1a^15b^14c^16d^8e^36f^39g^15h^21i^27j^32k^8l^4m^11n^6o^20 + 1a^13b^30c^9d^4e^9f^20g^5h^35i^19j^12k^29l^7m^21n^32o^28 + 1a^4b^17c^9d^21e^36f^9g^31h^10i^18j^23k^26l^32m^21n^34o^18 + 1a^12b^30c^28d^29e^9f^39g^32h^36i^28j^24k^10l^32m^22n^12o^10 + 1a^36b^13c^16d^16e^22f^20g^6h^5i^10j^15l^37m^30n^33 + 1a^10b^28c^20d^29e^22f^25g^29h^3i^39j^10k^31l^4m^38no^23 + 1a^16b^13c^29d^15e^13f^21g^23h^30i^18j^30k^15l^8m^5n^18o^6 + 1a^34b^26c^22d^6e^6f^6g^10h^13i^25j^12k^7l^20m^30n^6o^31 + 1a^25b^18c^38d^36e^34f^26g^3h^35i^16j^37k^25l^39m^37n^24o^39 + 1a^36b^11c^31d^21e^6f^9g^3h^7i^36j^7k^32l^35m^22n^14o^5 + 1a^27b^10c^37d^32e^22f^35g^3h^8i^27j^14k^4l^36m^39n^12o^7 + 1a^29b^35c^7d^32ef^6g^4h^14i^28j^17k^9l^22m^20n^33o^21 + 1a^26b^30c^4d^10e^37f^37g^6h^5i^2jk^19l^12m^31n^24o^33 + 1a^34b^6c^30d^20e^33f^18g^19h^17i^26j^5k^30l^3m^3n^35o^27 + 1a^4b^10c^6d^2e^13f^3g^32h^21i^15j^19k^10l^23mn^29o^25 + 1a^38b^8c^23de^22f^10g^8hi^11j^14k^6l^34m^27n^7o^39 + 1a^19b^31c^2d^16e^5f^37g^19h^25i^27j^16k^31l^23m^12n^29o^19 + 1a^38b^18c^28d^36e^26f^7g^34h^17i^32j^25k^29l^35m^26n^18o^14 + 1a^11b^25c^33d^15e^22f^3h^29i^4j^5k^6l^21m^12n^33o^11 
+ 1a^14b^39c^20d^4e^37f^7g^39h^9i^39j^27k^23l^34m^35n^18o^5 + 1a^25b^29c^6e^23f^5g^32h^3i^26j^2k^3l^25m^33n^7o^37 + 1a^7b^26c^22d^13f^25g^39h^6i^32j^11k^5l^6m^34n^33o^28 + 1a^13b^8c^5d^35e^4f^27g^18h^9i^39j^4l^32m^28n^9o^10 + 1a^3b^25c^32d^36e^18f^15g^22h^8i^32j^4k^22l^30m^11n^20o^38 + 1a^25b^6c^36d^8e^4f^6g^23h^28i^33j^7k^33l^38m^14n^27o^34 + 1a^26b^29c^2d^21e^39f^6g^11h^39i^4j^37k^31l^24m^32n^38o^26 + 1a^15b^17c^26d^16e^35f^34g^38h^23i^7j^14k^38l^5m^3n^36o^14 + 1a^23b^13c^39d^22e^9f^25g^34h^25i^21k^20l^30m^24n^39 + 1a^6b^17c^32d^18e^39f^13g^29h^34i^31j^28k^14l^9m^5n^19o^18 + 1ab^11c^13de^9f^21h^37i^23j^38l^5m^3n^24o^17 + 1a^24b^36c^36de^34f^35g^24h^34ij^10k^34l^38m^18n^30o^18 + 1a^20bc^12d^21e^11f^32g^35h^22i^30j^5k^11l^15m^34n^22o^39 + 1a^19b^33c^27d^19e^12f^4g^5h^19i^30j^39k^16m^39n^10o^13 + 1a^33b^31c^16d^35e^25f^4g^14h^5i^24j^19k^37l^8m^31n^8o + 1a^6b^38c^34d^6e^7f^26g^6h^20i^22j^21k^11l^26m^15n^27o^34 + 1a^39b^9c^30d^34e^8f^3g^10h^38i^23j^27k^29l^27m^10n^20o^20 + 1b^11c^19d^10e^23f^39g^9h^33i^34j^28k^5l^7m^28n^23o^25 
+ 1a^35b^31c^14d^20e^36f^30g^23h^9i^36j^16k^13l^27m^25n^6o^33 + 1a^22b^15c^15d^19e^27f^25g^31h^22i^18j^38k^30l^6m^30n^33o^8 + 1ab^30c^32d^39e^25f^23g^21h^24i^33j^7k^32l^28m^39n^34o^28 + 1a^15b^7c^8d^3e^19f^10h^2i^5j^32k^31l^34m^39n^11o^15 + 1a^11b^33c^33d^8e^28f^25g^17h^22ij^23l^27m^26n^2o^10 + 1a^29b^24c^7d^16f^29g^17h^18i^17j^26k^4l^3m^39no^4 + 1ab^23c^29d^2e^20f^24g^6h^30j^35k^38l^23mn^27o^36 + 1a^22b^18c^13e^22f^22g^5h^35i^21j^22k^20l^29m^25n^21o^29 + 1a^23b^39c^11d^33e^5f^16g^31h^25i^2j^9k^24l^2m^12n^16o^29 + 
1a^14b^27c^27d^3e^11f^7g^33h^7i^38j^23k^12l^33m^15n^21o^36 + 1a^30b^9c^26d^39e^2f^36h^33i^34j^37k^18l^18m^19n^23o^14 + 1a^10b^10c^12d^13e^26f^19g^38h^28i^32j^5k^6l^26m^39n^30o^28 + 1a^17b^18c^17d^20e^18f^12g^14h^10i^33j^15k^7l^26m^17n^36o + 1a^13b^9c^13d^11e^18fgh^33i^26j^33k^10l^7m^21n^10o^12 + 1a^13b^10c^7d^3e^16f^26g^30h^2i^3j^38k^20l^25m^2n^32o^15 + 1a^3b^32c^17d^27e^20f^37g^7h^33i^28j^18k^23l^29m^2n^28o^33 + 1a^4b^12c^19d^8e^18f^10g^33h^25i^37j^19k^39l^18m^36n^11o^32 + 1a^30b^13c^15d^13e^27f^14g^35h^36i^33j^23kl^18m^28n^32o^4 + 1a^13b^9c^8d^9e^19f^8g^30h^4i^27j^35k^11l^22m^31n^14o^28 + 1a^3b^11c^23d^35e^37f^4g^35h^28i^36j^10k^28l^11m^18n^2o^34.

## circuit.py
A simple library for creating, running, and visualizing arithmetic circuits as trees.

## brute_force.py
A semi-naive method of brute-forcing polynomial circuits (made obsolete by smart_force.py, but is still easier to understand). Modify main() for use.

## smart_force.py
A much more complex version of brute-force search, running many orders of magnitude faster. Saves output to .csv file that can be read by read_lib.py. Modify constraint constant for use.

## read_lib.py
Reads the output files generated by smart_force.py to print resulting polynomials and computation trees to the terminal. To use, add filename as argument (ex. 'python3 read_lib.py example.csv'). Prompted inputs represent the index of the desired polynomial (from first column of input file).

### See commments for details.
