(ns dplearn.layer.base
  (:use  [clj-tuple]
         [clojure.core.matrix]))

(defprotocol UpDown
  (defn idim [this])
  (defn odim [this])
  (defn invoke-up [this rvec cidx])
  (defn invoke-down [this ridx cvec])
  (defn sample-up [this vsample])
  (defn sample-down [this hsample])
  (defn gibbs-hvh [this hsample])
  (defn mini-batch [this k hsample])
  (defn contra-div! [this k lr vinput]))

(defrecord NeuronLayer [weights
                        fn-activition-up  fn-activition-down
                        fn-sample-up  fn-sample-down])

(extend-type NeuronLayer UpDown

  (defn idim [this]
    (row-count (:weights this)))

  (defn odim [this]
    (column-count (:weights this)))

  (defn invoke-up [this rvec cidx]
    (apply (:fn-activition-up this)
           (dot rvec (get-column (:weights this) cidx))))

  (defn invoke-down [this ridx cvec]
    (apply (:fn-activition-down this)
           (dot (get-row (:weights this) ridx) cvec)))

  (defn sample-up [this vsample]
    (let [hmeans (map #(invoke-up this vsample %) (range (odim this)))
          hsample (map (:fn-sample-up this) hmeans)]
      (tuple hmeans hsample)))

  (defn sample-down [this hsample]
    (let [vmeans (map #(invoke-down this % hsample) (range (idim this)))
          vsample (map (:fn-sample-down this) vmeans)]
      (tuple vmeans vsample)))

  (defn gibbs-hvh [this hsample]
    (let [[nvmeans nvsample] (sample-down hsample)
          [nhmeans nhsample] (sample-up nvsample)]
      (tuple nvmeans nvsample nhmeans nhsample)))

  (defn mini-batch [this k hsample]
    (loop [i k
           [nvmeans nvsample nhmeans nhsample] [nil nil nil hsample]]
      (if (zero? i)
        (tuple nvmeans nhsample)
        (recur (dec i) (gibbs-hvh this nhsample)))))

  (defn contra-div! [this k lr vsample])
    (let [[hmeans hsample] (sample-up this vinput)
          [nvmeans nhsample] (mini-batch this k hsample)
          mw (:weights this)
          mp (mmul (column-matrix hmeans) (row-matrix vinput))
          mn (mmul (column-matrix nvmeans) (row-matrix nhsample))]
      (do
        (add! mw (scale mp lr))
        (add! mw (scale mn (- lr)))
        this)))

(defmacro defnnkind [docstring kind-name & {:as args}]
  (let [kind-name#                (symbol kind-name)
        weights-name#             (symbol (str "weights-" kind-name))
        invoke-up-name#           (symbol (str "invoke-up-" kind-name))
        invoke-down-name#         (symbol (str "invoke-down-" kind-name))
        fan-in#                   (:fan-in args)
        fan-out#                  (:fan-out args)
        activition-up#            (or (:activition-up args) (fn [ctx] ctx))
        activition-down#          (or (:activition-down args) (fn [ctx] ctx))]
    `(do
       (def ~kind-name#
         (fn [] )))))
