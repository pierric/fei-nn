{-# LANGUAGE TemplateHaskell #-}
module MXNet.NN.Utils.HMap (hm) where

import Language.Haskell.TH
import Language.Haskell.TH.Quote
import Text.ParserCombinators.ReadP
import qualified Data.Char as Char
import Language.Haskell.Meta.Parse
import qualified MXNet.Core.Base.HMap as M

hm :: QuasiQuoter
hm = QuasiQuoter { quoteExp = parseHMap
                 , quotePat = undefined
                 , quoteType= undefined
                 , quoteDec = undefined }

parseHMap :: String -> Q Exp
parseHMap src = case pr of 
                  Left e -> fail (show e)
                  Right lst -> foldr makeKV nilE lst
  where
    pr = parse items src
    nilE = [| M.nil |]
    makeKV :: Item -> Q Exp ->  Q Exp
    makeKV (Item s e) m = appTypeE [| M.add |] (litT $ strTyLit s) `appE` (return e) `appE` m
 
data Item = Item { _item_key :: String, _item_val :: Exp }
  deriving Show

data ParseError = CannotParse | Ambiguous
  deriving Show

items :: ReadP [Item]
items = (skipSpaces >> eof >> return []) <++ sepBy item (char ',')
  where
    item = do 
        skipSpaces
        k <- many1 (satisfy $ not . Char.isSpace)
        skipSpaces
        _ <- string ":="
        skipSpaces
        v <- many1 get
        case parseExp v of 
            Left _  -> pfail
            Right e -> return $ Item k e

parse :: ReadP [Item] -> String -> Either ParseError [Item]
parse p s = let cans = readP_to_S p s
            in case filter (\(_,left) -> length left == 0) cans of
                 [] -> Left CannotParse
                 [(r,_)] -> Right r
                 _  -> Left Ambiguous