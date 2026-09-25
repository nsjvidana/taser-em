use proc_macro::{TokenStream};
use proc_macro2::{TokenStream as TokenStream2, TokenTree as TokenTree2, Ident as Ident2};
use quote::{quote, ToTokens};
use syn::{parse_macro_input, Ident};
use syn::spanned::Spanned;

/// Clones a function with specified identifiers replaced. Replace identifiers by specifying their
/// name as an attribute equal to another identifier name.
///
/// Must include a `suffix` attribute that's a string literal. `suffix` will be appended to the end
/// of the new function clone name with an underscore (see example below).
///
/// This attribute can be stacked to create multiple clones.
///
/// # Example
/// The following function:
/// ```
/// #[replace_idents(suffix = "cool_suffix1", axis = z, axis1 = x, axis2 = y)]
/// #[replace_idents(suffix = "cool_suffix2", axis = z, axis1 = y, axis2 = x)]
/// fn axis_stuff(b: UVec3) -> f32 {
///     b.axis * (b.axis2 - b.axis1)
/// }
/// ```
/// turns into:
///
/// ```
/// #[replace_idents(suffix = "cool_suffix2", axis = z, axis1 = y, axis2 = x)]
/// fn axis_stuff(b: UVec3) -> f32 {
///     b.axis * (b.axis2 - b.axis1)
/// }
///
/// fn axis_stuff_cool_suffix1(b: UVec3) -> f32 {
///     b.z * (b.y - b.x)
/// }
/// ```
#[proc_macro_attribute]
pub fn replace_idents(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = syn::parse_macro_input!(item as syn::ItemFn);

    let mut fn_name_suffix = None;
    let mut targets = vec![];
    let mut replacements = vec![];
    let attr_parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("suffix") {
            let suffix: syn::LitStr = meta.value()?.parse()?;
            fn_name_suffix = Some(suffix.value());
        } else if let Some(ident) = meta.path.get_ident() {
            targets.push(ident.to_string());
            let replacement_ident: Ident = meta.value()?.parse()?;
            replacements.push(replacement_ident.to_string());
        } else {
            return Err(meta.error(
                format!("Unsupported attribute path: {}", meta.path.clone().into_token_stream())
            ));
        }
        Ok(())
    });
    parse_macro_input!(attr with attr_parser);

    if fn_name_suffix.is_none() {
        return syn::Error::new(func.span(), "A \"suffix\" attribute is required")
            .to_compile_error().into();
    } else if targets.is_empty() || replacements.is_empty() {
        return syn::Error::new(func.span(), "Expected at least one identifier to replace")
            .to_compile_error().into();
    }
    let fn_name_suffix = fn_name_suffix.unwrap();

    let og_name = func.sig.ident.to_string();
    let new_name = format!("{og_name}_{fn_name_suffix}");
    let new_ident = Ident::new(&new_name, func.sig.ident.span());
    let mut func_clone = func.clone();
    func_clone.sig.ident = new_ident;

    let mut new_block = replace_idents_in_stream(
        func_clone.block.into_token_stream(),
        targets[0].as_str(),
        replacements[0].as_str()
    );
    for (target, replacement) in targets.iter()
        .zip(replacements.iter())
        .skip(1)
    {
        new_block = replace_idents_in_stream(
            new_block,
            target.as_str(),
            replacement.as_str()
        );
    }
    func_clone.block = syn::parse2(new_block).unwrap();

    func_clone.attrs.retain(|a| {
        !a.path().is_ident("replace_idents")
    });

    let n_self_attrs = func.attrs.iter()
        .filter(|a| a.path().is_ident("replace_idents"))
        .count();

    if n_self_attrs != 0 {
        TokenStream::from(quote! {
            #func
            #func_clone
        })
    }
    else {
        TokenStream::from(quote! {
            #func_clone
        })
    }
}

fn replace_idents_in_stream(stream: TokenStream2, target: &str, replacement: &str) -> TokenStream2 {
    let mut new_stream = TokenStream2::new();

    for tt in stream {
        match tt {
            TokenTree2::Ident(ref ident) if ident == target => {
                let new_ident = Ident2::new(replacement, ident.span());
                new_stream.extend(quote! { #new_ident });
            }
            TokenTree2::Group(group) => {
                let inner_stream = replace_idents_in_stream(group.stream(), target, replacement);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), inner_stream);
                new_group.set_span(group.span());
                new_stream.extend(quote! { #new_group });
            }
            other => new_stream.extend(quote! { #other })
        }
    }

    new_stream
}